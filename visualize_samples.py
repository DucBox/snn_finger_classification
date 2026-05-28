import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

MATRIX_ROWS = 27
MATRIX_COLS = 15
OUTPUT_DIR  = Path("outputs/sample_images")
PARQUET     = Path("outputs/full_data_set.parquet")

FINGERS = ["thumb", "index", "middle", "ring", "little"]
TASKS   = ["TAP", "DRAG", "SCROLL"]
N_PER_GROUP = 2   # samples per (finger × task) combination


def save_heatmap(matrix: np.ndarray, title: str, filepath: Path):
    fig, ax = plt.subplots(figsize=(3, 5))
    im = ax.imshow(matrix, cmap="hot", aspect="auto",
                   vmin=0, vmax=matrix.max() or 1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (15 cols)")
    ax.set_ylabel("y (27 rows)")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(PARQUET)
    saved = 0

    for finger in FINGERS:
        for task in TASKS:
            subset = df.filter(
                (pl.col("Finger") == finger) & (pl.col("Task") == task)
            )
            if len(subset) == 0:
                continue

            # pick samples spread across different participants
            picks = (
                subset
                .group_by("Participant")
                .agg(pl.first("BlobImgFlattened"), pl.first("Handedness"))
                .sort("Participant")
                .head(N_PER_GROUP)
            )

            for row in picks.iter_rows(named=True):
                flat   = row["BlobImgFlattened"]
                matrix = np.array(flat, dtype=float).reshape(MATRIX_ROWS, MATRIX_COLS)

                p    = row["Participant"]
                hand = row["Handedness"]
                name = f"{finger}_{task}_P{p}_{hand}"
                title = f"Finger: {finger} | Task: {task}\nP{p} ({hand})"

                save_heatmap(matrix, title, OUTPUT_DIR / f"{name}.png")
                saved += 1

    print(f"Saved {saved} images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
