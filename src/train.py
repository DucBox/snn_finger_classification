from __future__ import annotations

import numpy as np
import polars as pl
import tensorflow as tf

from src.utils import set_seeds, split_participants, SEED

# ── Label configs ─────────────────────────────────────────────────────────────

FINGER_MAP = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}
HAND_MAP   = {"left": 0, "right": 1}

TASK_CONFIGS: dict[str, dict] = {
    "thumb_lr":  {"filter_fingers": ["thumb"], "n_classes": 2},
    "index_lr":  {"filter_fingers": ["index"], "n_classes": 2},
    "5fingers":  {"filter_fingers": None,       "n_classes": 5},
    "10fingers": {"filter_fingers": None,       "n_classes": 10},
}


def _make_labels(df: pl.DataFrame, task: str) -> np.ndarray:
    if task in ("thumb_lr", "index_lr"):
        return np.array([HAND_MAP[h] for h in df["Handedness"].to_list()], dtype=np.int32)
    if task == "5fingers":
        return np.array([FINGER_MAP[f] for f in df["Finger"].to_list()], dtype=np.int32)
    return np.array(
        [FINGER_MAP[f] * 2 + HAND_MAP[h]
         for f, h in zip(df["Finger"].to_list(), df["Handedness"].to_list())],
        dtype=np.int32,
    )


# ── Device setup ──────────────────────────────────────────────────────────────

def setup_device(device_arg: str) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if device_arg == "cpu" or (device_arg == "auto" and not gpus):
        tf.config.set_visible_devices([], "GPU")
        return "CPU"
    if gpus:
        # allow memory growth to avoid grabbing all VRAM upfront
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return f"GPU (x{len(gpus)})"
    return "CPU"


# ── Validation ────────────────────────────────────────────────────────────────

def validate_data(df: pl.DataFrame, train_ids: list[str], test_ids: list[str], task: str):
    print("[VALIDATE] Checking data integrity ...")

    required = {"Participant", "Finger", "Handedness", "Task", "Timestamp", "BlobResized8x8"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    print(f"  Columns OK")

    null_counts = df.select([pl.col(c).is_null().sum().alias(c) for c in required])
    for col, cnt in zip(null_counts.columns, null_counts.row(0)):
        assert cnt == 0, f"Column '{col}' has {cnt} null values"
    print("  No nulls OK")

    lengths = df["BlobResized8x8"].list.len()
    assert lengths.min() == 64 and lengths.max() == 64, \
        f"BlobResized8x8 length range [{lengths.min()}, {lengths.max()}] != 64"
    print("  Feature shape OK: all 64 values")

    all_vals = np.array(df["BlobResized8x8"].explode().to_list())
    assert all_vals.min() >= 0 and all_vals.max() <= 255, \
        f"Value range [{all_vals.min()}, {all_vals.max()}] outside [0, 255]"
    print(f"  Value range OK: [{int(all_vals.min())}, {int(all_vals.max())}]")

    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"Participant overlap: {overlap}"
    print(f"  Split OK: train={len(train_ids)} participants | test={len(test_ids)} participants")

    cfg = TASK_CONFIGS[task]
    df_t = df.filter(pl.col("Finger").is_in(cfg["filter_fingers"])) \
        if cfg["filter_fingers"] else df
    if task in ("thumb_lr", "index_lr"):
        dist = df_t.group_by("Handedness").agg(pl.len()).sort("Handedness")
        print(f"  Class distribution ({task}):")
        for r in dist.iter_rows(named=True):
            print(f"    {r['Handedness']}: {r['len']:,}")
    elif task == "5fingers":
        dist = df_t.group_by("Finger").agg(pl.len()).sort("Finger")
        print(f"  Class distribution (5fingers):")
        for r in dist.iter_rows(named=True):
            print(f"    {r['Finger']}: {r['len']:,}")
    else:
        dist = (df_t
                .with_columns((pl.col("Finger") + "_" + pl.col("Handedness")).alias("label"))
                .group_by("label").agg(pl.len()).sort("label"))
        print(f"  Class distribution (10fingers):")
        for r in dist.iter_rows(named=True):
            print(f"    {r['label']}: {r['len']:,}")

    print("[VALIDATE] All checks passed.\n")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_tiny_model(n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="TinyNet")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    data_path:  str,
    task:       str   = "5fingers",
    epochs:     int   = 5,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    device_arg: str   = "auto",
):
    set_seeds(SEED)
    device = setup_device(device_arg)
    print(f"Device: {device}")

    # Step 1: Load + sort by timestamp
    print(f"\n[STEP 1/4] Loading data: {data_path}")
    df = pl.read_parquet(data_path).sort("Timestamp")
    print(f"[STEP 1/4] Done. {len(df):,} samples.\n")

    # Step 2: Validate
    print("[STEP 2/4] Validating ...")
    all_ptcp = df["Participant"].unique().to_list()
    train_ids, test_ids = split_participants(all_ptcp)
    validate_data(df, train_ids, test_ids, task)

    # Step 3: Build tf.data datasets
    print(f"[STEP 3/4] Building datasets for task='{task}' ...")
    cfg = TASK_CONFIGS[task]

    df_train = df.filter(pl.col("Participant").is_in(train_ids))
    df_test  = df.filter(pl.col("Participant").is_in(test_ids))
    if cfg["filter_fingers"]:
        df_train = df_train.filter(pl.col("Finger").is_in(cfg["filter_fingers"]))
        df_test  = df_test.filter(pl.col("Finger").is_in(cfg["filter_fingers"]))

    X_train = np.array(df_train["BlobResized8x8"].to_list(), dtype=np.float32) / 255.0
    y_train = _make_labels(df_train, task)
    X_test  = np.array(df_test["BlobResized8x8"].to_list(),  dtype=np.float32) / 255.0
    y_test  = _make_labels(df_test,  task)

    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_train, y_train))
                .shuffle(10_000, seed=SEED)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    test_ds  = (tf.data.Dataset
                .from_tensor_slices((X_test, y_test))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    # Step 4: Build + train model
    print(f"\n[STEP 4/4] Building and training model ...")
    model = build_tiny_model(cfg["n_classes"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print()

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=2,
    )
    print("\n[STEP 4/4] Done.")
