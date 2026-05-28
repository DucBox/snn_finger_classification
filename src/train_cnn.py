from __future__ import annotations

import os
import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import RMSprop

from src.utils import set_seeds, split_participants, SEED

MATRIX_ROWS = 27
MATRIX_COLS  = 15
HAND_MAP     = {"left": 0, "right": 1}


def build_cnn() -> tf.keras.Model:
    return tf.keras.Sequential([
        Conv2D(64, (3, 3), padding="same", activation="relu",
               input_shape=(MATRIX_ROWS, MATRIX_COLS, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.5),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(256, activation="relu", kernel_initializer=glorot_uniform()),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ], name="CapFingerId_CNN")


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
    output_model: str = "outputs/model_cnn.h5",
    epochs:       int   = 100,
    batch_size:   int   = 50,
    device_arg:   str   = "auto",
):
    set_seeds(SEED)
    print(f"Device: {_setup_device(device_arg)}")

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

    # CNN input: BlobImgFlattened reshaped to (27, 15, 1), raw values (no normalization)
    X_train = (np.array(df_train["BlobImgFlattened"].to_list(), dtype=np.float32)
               .reshape(-1, MATRIX_ROWS, MATRIX_COLS, 1))
    X_test  = (np.array(df_test["BlobImgFlattened"].to_list(),  dtype=np.float32)
               .reshape(-1, MATRIX_ROWS, MATRIX_COLS, 1))
    y_train = np.array([HAND_MAP[h] for h in df_train["Handedness"].to_list()], dtype=np.float32)
    y_test  = np.array([HAND_MAP[h] for h in df_test["Handedness"].to_list()],  dtype=np.float32)

    print(f"[STEP 2/4] X_train {X_train.shape}  X_test {X_test.shape}")
    left_train  = int((y_train == 0).sum())
    right_train = int((y_train == 1).sum())
    left_test   = int((y_test == 0).sum())
    right_test  = int((y_test == 1).sum())
    print(f"  Train  left={left_train:,}  right={right_train:,}")
    print(f"  Test   left={left_test:,}   right={right_test:,}\n")

    print("[STEP 3/4] Building model ...")
    model = build_cnn()
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8),
        metrics=["accuracy"],
    )
    model.summary()
    print()

    print("[STEP 4/4] Training ...")
    os.makedirs(os.path.dirname(os.path.abspath(output_model)), exist_ok=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint(output_model, save_best_only=True,
                            monitor="val_accuracy", mode="max"),
            EarlyStopping(monitor="val_accuracy", patience=5, verbose=0),
        ],
        verbose=2,
    )

    best = tf.keras.models.load_model(output_model)
    loss, acc = best.evaluate(X_test, y_test, verbose=0)
    print(f"\n[STEP 4/4] Done.  Best model  |  accuracy={acc:.4f}  loss={loss:.4f}")
