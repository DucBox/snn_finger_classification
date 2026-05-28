import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET    = Path("outputs/snn_finger_processed_data.parquet")
OUTPUT_DIR = Path("outputs/snn_sample_images")
FINGERS    = ["thumb", "index", "middle", "ring", "little"]
TASKS      = ["TAP", "DRAG", "SCROLL"]
N_PER_GROUP = 3


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(PARQUET)

    for finger in FINGERS:
        for task in TASKS:
            subset = df.filter(
                (pl.col("Finger") == finger) & (pl.col("Task") == task)
            )
            if len(subset) == 0:
                continue

            picks = (
                subset
                .group_by("Participant")
                .agg(
                    pl.first("BlobResized8x8"),
                    pl.first("BlobCrop"),
                    pl.first("Handedness"),
                )
                .sort("Participant")
                .head(N_PER_GROUP)
            )

            rows = picks.iter_rows(named=True)
            samples = list(rows)
            n = len(samples)

            fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3))
            if n == 1:
                axes = [axes]

            for ax, row in zip(axes, samples):
                img8 = np.array(row["BlobResized8x8"], dtype=np.uint8).reshape(8, 8)
                ax.imshow(img8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.set_title(f"P{row['Participant']}\n{row['Handedness']}", fontsize=7)
                ax.axis("off")

            fig.suptitle(f"{finger.capitalize()} — {task}", fontsize=9, y=1.02)
            plt.tight_layout()
            out_path = OUTPUT_DIR / f"{finger}_{task}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
