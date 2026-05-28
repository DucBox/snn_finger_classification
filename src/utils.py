from __future__ import annotations

import math
import os
import random
import numpy as np

SEED = 42

ALL_FINGERS = ["thumb", "index", "middle", "ring", "little"]

# Each task defines which fingers to include and how to build the label.
# "fingers": subset of ALL_FINGERS (or ALL_FINGERS for all)
# "label":   one of "handedness" | "finger" | "finger_hand" | "thumb_others"
TASKS = {
    "thumb_lr":     {"fingers": ["thumb"],          "label": "handedness"},
    "index_lr":     {"fingers": ["index"],          "label": "handedness"},
    "hand_lr":      {"fingers": ALL_FINGERS,        "label": "handedness"},
    "5fingers":     {"fingers": ALL_FINGERS,        "label": "finger"},
    "10fingers":    {"fingers": ALL_FINGERS,        "label": "finger_hand"},
    "thumb_index":  {"fingers": ["thumb", "index"], "label": "finger"},
    "thumb_others": {"fingers": ALL_FINGERS,        "label": "thumb_others"},
}


def _extract_raw_labels(df, task: str):
    """Filter df to task's fingers and return (filtered_df, raw_string_labels)."""
    import polars as pl
    cfg = TASKS[task]
    df = df.filter(pl.col("Finger").is_in(cfg["fingers"]))
    ltype = cfg["label"]
    if ltype == "handedness":
        raw = df["Handedness"].to_list()
    elif ltype == "finger":
        raw = df["Finger"].to_list()
    elif ltype == "finger_hand":
        raw = [f"{f}_{h}" for f, h in zip(df["Finger"].to_list(), df["Handedness"].to_list())]
    elif ltype == "thumb_others":
        raw = ["thumb" if f == "thumb" else "others" for f in df["Finger"].to_list()]
    else:
        raise ValueError(f"Unknown label type: {ltype}")
    return df, raw


def build_label_map(df_full, task: str) -> dict:
    """Build canonical label_map from the full (unsplit) dataset."""
    _, raw = _extract_raw_labels(df_full, task)
    return {lbl: i for i, lbl in enumerate(sorted(set(raw)))}


def apply_labels(df, task: str, label_map: dict):
    """
    Filter df to task's fingers and encode labels using the provided label_map.
    Returns (filtered_df, y_int).
    """
    df, raw = _extract_raw_labels(df, task)
    y_int = np.array([label_map[l] for l in raw], dtype=np.int32)
    return df, y_int


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def split_participants(all_participants: list[str], split_factor: float = 0.8):
    sorted_ptcp = sorted(all_participants, key=lambda x: int(x))
    n_train = math.floor(len(sorted_ptcp) * split_factor)
    return sorted_ptcp[:n_train], sorted_ptcp[n_train:]
