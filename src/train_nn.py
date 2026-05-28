from __future__ import annotations

import os
import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, BatchNormalization, Activation, Concatenate, Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics as km
from sklearn.preprocessing import MinMaxScaler

from src.utils import set_seeds, split_participants, SEED

HAND_MAP = {"left": 0, "right": 1}


def build_tiny_nn(n_classes: int = 2) -> tf.keras.Model:
    """Small model for quick dev/test runs."""
    input_blob = Input(shape=(64,), name="input_blob")
    input_area = Input(shape=(1,),  name="input_area")

    x = Dense(32, activation="relu")(input_blob)
    y = Dense(8,  activation="relu")(input_area)

    z = Concatenate()([x, y])
    z = Dense(16, activation="relu")(z)

    output = Dense(n_classes, activation="softmax")(z)
    return Model(inputs=[input_blob, input_area], outputs=output, name="TinyNN")


def build_full_nn(n_classes: int = 2) -> tf.keras.Model:
    """Full dual-input architecture from the paper notebook."""
    input_blob = Input(shape=(64,), name="input_blob")
    input_area = Input(shape=(1,),  name="input_area")

    x = Dense(128)(input_blob)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    y = Dense(128)(input_area)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    z = Concatenate()([x, y])
    z = Dense(64)(z);  z = BatchNormalization()(z);  z = Activation("relu")(z)
    z = Dense(32)(z);  z = BatchNormalization()(z);  z = Activation("relu")(z)
    z = Dense(16)(z);  z = BatchNormalization()(z);  z = Activation("relu")(z)

    output = Dense(n_classes, activation="softmax")(z)
    return Model(inputs=[input_blob, input_area], outputs=output, name="OptimizedNN")


_MODELS = {"tiny": build_tiny_nn, "full": build_full_nn}


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
    output_model: str  = "outputs/model_nn.h5",
    model_type:   str  = "full",
    epochs:       int  = 100,
    batch_size:   int  = 16,
    device_arg:   str  = "auto",
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

    print("[STEP 3/4] Normalizing features (MinMaxScaler) ...")
    scaler_feat = MinMaxScaler()
    scaler_area = MinMaxScaler()
    X_feat_train = scaler_feat.fit_transform(X_feat_train).astype(np.float32)
    X_feat_test  = scaler_feat.transform(X_feat_test).astype(np.float32)
    X_area_train = scaler_area.fit_transform(X_area_train).astype(np.float32)
    X_area_test  = scaler_area.transform(X_area_test).astype(np.float32)

    y_train = to_categorical(y_train_int, num_classes=2)
    y_test  = to_categorical(y_test_int,  num_classes=2)

    print(f"  X_feat {X_feat_train.shape}  X_area {X_area_train.shape}")
    print(f"  Train  left={int((y_train_int==0).sum()):,}  right={int((y_train_int==1).sum()):,}\n")

    print("[STEP 4/4] Building and training model ...")
    model = _MODELS[model_type](n_classes=2)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[km.BinaryAccuracy(name="accuracy")],
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
                            monitor="val_accuracy", mode="max"),
            EarlyStopping(monitor="val_accuracy", patience=50, verbose=0),
        ],
        verbose=2,
    )

    best = tf.keras.models.load_model(output_model)
    loss, acc = best.evaluate([X_feat_test, X_area_test], y_test, verbose=0)
    print(f"\n[STEP 4/4] Done.  Best model  |  accuracy={acc:.4f}  loss={loss:.4f}")
