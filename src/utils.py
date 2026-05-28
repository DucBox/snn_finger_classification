from __future__ import annotations

import os
import random
import numpy as np

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set all relevant seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def split_participants(all_participants: list[str], split_factor: float = 0.8):
    """
    Sort participants numerically, split into train/test by participant ID.
    Returns (train_ids, test_ids).
    """
    import math
    sorted_ptcp = sorted(all_participants, key=lambda x: int(x))
    n_train = math.floor(len(sorted_ptcp) * split_factor)
    return sorted_ptcp[:n_train], sorted_ptcp[n_train:]
