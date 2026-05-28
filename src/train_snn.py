from __future__ import annotations

import os
import numpy as np
import polars as pl
import tensorflow as tf
import keras_spiking
from tensorflow.keras.layers import (
    TimeDistributed, Dense, Concatenate, GlobalAveragePooling1D, Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from src.utils import set_seeds, split_participants, SEED

HAND_MAP = {"left": 0, "right": 1}
N_STEPS  = 5
DT       = 0.3


def build_tiny_snn(
    n_classes: int   = 2,
    n_steps:   int   = N_STEPS,
    dt:        float = DT,
) -> tf.keras.Model:
    """Small spiking model for quick dev/test runs."""
    input_blob = Input(shape=(n_steps, 64), name="input_blob")
    input_area = Input(shape=(n_steps, 1),  name="input_area")

    x = TimeDistributed(Dense(32))(input_blob)
    x = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(x)

    y = TimeDistributed(Dense(8))(input_area)
    y = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(y)

    z = Concatenate(axis=-1)([x, y])
    z = TimeDistributed(Dense(16))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(z)

    z      = GlobalAveragePooling1D()(z)
    output = Dense(n_classes, activation="softmax")(z)

    return Model(inputs=[input_blob, input_area], outputs=output, name="TinySNN")


def build_full_snn(
    n_classes: int   = 2,
    n_steps:   int   = N_STEPS,
    dt:        float = DT,
) -> tf.keras.Model:
    """Full spiking architecture from the paper notebook."""
    input_blob = Input(shape=(n_steps, 64), name="input_blob")
    input_area = Input(shape=(n_steps, 1),  name="input_area")

    x = TimeDistributed(Dense(128))(input_blob)
    x = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(x)

    y = TimeDistributed(Dense(128))(input_area)
    y = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(y)

    z = Concatenate(axis=-1)([x, y])
    z = TimeDistributed(Dense(64))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(z)
    z = TimeDistributed(Dense(32))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(z)
    z = TimeDistributed(Dense(16))(z)
    z = keras_spiking.SpikingActivation("relu", dt=dt, spiking_aware_training=True)(z)

    z      = GlobalAveragePooling1D()(z)
    output = Dense(n_classes, activation="softmax")(z)

    return Model(inputs=[input_blob, input_area], outputs=output, name="OptimizedSNN")


_MODELS = {"tiny": build_tiny_snn, "full": build_full_snn}


def _setup_device(device_arg: str) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if device_arg == "cpu" or (device_arg == "auto" and not gpus):
        tf.config.set_visible_devices([], "GPU")
        return "CPU"
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return f"GPU (x{len(gpus)})"


def run(
    data_path:    str,
    output_model: str   = "outputs/model_snn.h5",
    model_type:   str   = "full",
    epochs:       int   = 70,
    batch_size:   int   = 16,
    n_steps:      int   = N_STEPS,
    dt:           float = DT,
    device_arg:   str   = "auto",
):
    assert model_type in _MODELS, f"model_type must be one of {list(_MODELS)}"
    set_seeds(SEED)
    print(f"Device: {_setup_device(device_arg)}")
    print(f"Model type: {model_type}")

    print(f"\n[STEP 1/4] Loading data: {data_path}")
    df = pl.read_parquet(data_path).sort("Timestamp")
    df = df.filter(pl.col("Finger").is_in(["thumb", "index"]))
    print(f"[STEP 1/4] Done. {len(df):,} samples (thumb + index).\n")

    print("[STEP 2/4] Splitting participants ...")
    all_ptcp = df["Participant"].unique().to_list()
    train_ids, test_ids = split_participants(all_ptcp)
    assert not (set(train_ids) & set(test_ids)), "Participant overlap detected"
    print(f"  Train: P{train_ids[0]} → P{train_ids[-1]}  ({len(train_ids)} participants)")
    print(f"  Test:  P{test_ids[0]}  → P{test_ids[-1]}   ({len(test_ids)} participants)\n")

    df_train = df.filter(pl.col("Participant").is_in(train_ids))
    df_test  = df.filter(pl.col("Participant").is_in(test_ids))

    X_feat_train = np.array(df_train["BlobResized8x8"].to_list(), dtype=np.float64)
    X_feat_test  = np.array(df_test["BlobResized8x8"].to_list(),  dtype=np.float64)
    X_area_train = df_train["Area"].to_numpy().reshape(-1, 1)
    X_area_test  = df_test["Area"].to_numpy().reshape(-1, 1)
    y_train_int  = np.array([HAND_MAP[h] for h in df_train["Handedness"].to_list()], dtype=np.int32)
    y_test_int   = np.array([HAND_MAP[h] for h in df_test["Handedness"].to_list()],  dtype=np.int32)

    print(f"[STEP 3/4] Normalizing (MinMaxScaler) and tiling to n_steps={n_steps} ...")
    scaler_feat = MinMaxScaler()
    scaler_area = MinMaxScaler()
    X_feat_train = scaler_feat.fit_transform(X_feat_train).astype(np.float32)
    X_feat_test  = scaler_feat.transform(X_feat_test).astype(np.float32)
    X_area_train = scaler_area.fit_transform(X_area_train).astype(np.float32)
    X_area_test  = scaler_area.transform(X_area_test).astype(np.float32)

    X_feat_train = np.tile(X_feat_train[:, None, :], (1, n_steps, 1))
    X_feat_test  = np.tile(X_feat_test[:, None, :],  (1, n_steps, 1))
    X_area_train = np.tile(X_area_train[:, None, :], (1, n_steps, 1))
    X_area_test  = np.tile(X_area_test[:, None, :],  (1, n_steps, 1))

    y_train = to_categorical(y_train_int, num_classes=2)
    y_test  = to_categorical(y_test_int,  num_classes=2)

    print(f"  X_feat {X_feat_train.shape}  X_area {X_area_train.shape}\n")

    print("[STEP 4/4] Building and training SNN model ...")
    model = _MODELS[model_type](n_classes=2, n_steps=n_steps, dt=dt)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    model.summary()
    print()

    os.makedirs(os.path.dirname(os.path.abspath(output_model)), exist_ok=True)
    model.fit(
        [X_feat_train, X_area_train], y_train,
        validation_data=([X_feat_test, X_area_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint(output_model, save_best_only=True,
                            monitor="val_categorical_accuracy", mode="max"),
        ],
        verbose=2,
    )

    best = tf.keras.models.load_model(
        output_model,
        custom_objects={"SpikingActivation": keras_spiking.SpikingActivation},
    )
    loss, acc = best.evaluate([X_feat_test, X_area_test], y_test, verbose=0)
    print(f"\n[STEP 4/4] Done.  Best model  |  accuracy={acc:.4f}  loss={loss:.4f}")
