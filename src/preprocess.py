from __future__ import annotations

import os
import sys
import numpy as np
import polars as pl
from tqdm import tqdm
from multiprocessing import Pool

sys.setrecursionlimit(10000)

THRESHOLD = 30
MATRIX_ROWS = 27
MATRIX_COLS = 15
CHUNK_SIZE = 1000


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_raw_data(data_path: str) -> pl.DataFrame:
    txt_files = sorted(f for f in os.listdir(data_path) if f.endswith(".txt"))
    frames = []
    for filename in tqdm(txt_files, desc="  Reading files", unit="file"):
        df_temp = pl.read_csv(
            os.path.join(data_path, filename),
            has_header=False,
            new_columns=["Participant", "Handedness", "Finger", "Task", "Ignore", "MatrixStr"],
            separator=";",
            infer_schema_length=0,
        )
        frames.append(df_temp)
    return pl.concat(frames).drop("Ignore")


# ── Parsing (module-level for multiprocessing pickling) ───────────────────────

def _parse_row(s: str) -> tuple[int, list | None]:
    """Parse one MatrixStr into (timestamp_ms, flat_matrix_405_ints).
    Returns (-1, None) on any error."""
    parts = s.replace("\n", "").split(",")
    if len(parts) != 408:
        return -1, None
    try:
        timestamp = int(str(parts[0]) + str(parts[1][:3]))
    except (ValueError, IndexError):
        return -1, None

    arr = np.array(parts[2:407]).reshape(MATRIX_ROWS, MATRIX_COLS)
    arr[arr == ""] = "0"
    arr[arr == "-"] = "0"
    try:
        return timestamp, arr.astype(int).flatten().tolist()
    except ValueError:
        return -1, None


# ── Blob detection (module-level for multiprocessing pickling) ────────────────

def _dfs(matrix, x, y, found):
    if (
        0 < x < len(matrix[0])
        and 0 < y < len(matrix)
        and matrix[y][x] > THRESHOLD
        and (x, y) not in found
    ):
        found.append((x, y))
        _dfs(matrix, x + 1, y, found)
        _dfs(matrix, x - 1, y, found)
        _dfs(matrix, x, y + 1, found)
        _dfs(matrix, x, y - 1, found)


def _get_blobs(matrix):
    blobs = []
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            found = []
            _dfs(matrix, x, y, found)
            if found:
                xs = sorted(found, key=lambda p: p[0])
                ys = sorted(found, key=lambda p: p[1])
                x_min, x_max = xs[0][0], xs[-1][0]
                y_min, y_max = ys[0][1], ys[-1][1]
                bbox = (x_min - 1, x_max + 1, y_min - 1, y_max + 1)
                if bbox not in blobs and (x_max - x_min) * (y_max - y_min) > 1:
                    blobs.append(bbox)
    return blobs


def process_blob(flat: list | None) -> list | None:
    """Run blob detection on a flat 405-int matrix.
    Returns blob image as flat list of 405 ints aligned to top-left, or None if not exactly 1 blob."""
    if flat is None:
        return None
    matrix = np.array(flat).reshape(MATRIX_ROWS, MATRIX_COLS)
    blobs = _get_blobs(matrix)
    if len(blobs) != 1:
        return None
    c = blobs[0]
    blob = matrix[c[2]:c[3], c[0]:c[1]]
    img = np.pad(
        blob,
        ((0, MATRIX_ROWS - blob.shape[0]), (0, MATRIX_COLS - blob.shape[1])),
        mode="constant",
        constant_values=0,
    )
    return img.flatten().tolist()


# ── Full pipeline ─────────────────────────────────────────────────────────────

BATCH_SIZE = 50_000


def run(data_path: str, output_path: str, n_workers: int = 4) -> pl.DataFrame:
    import tempfile, shutil

    # Step 1: Load
    print(f"\n[STEP 1/3] Loading raw data from: {data_path}")
    df = load_raw_data(data_path)
    n_total = len(df)
    print(f"[STEP 1/3] Done. {n_total:,} raw samples loaded.\n")

    # Step 2: Chunked parse + PAUSE filter + blob detection
    # Slicing df in batches avoids ever holding full raw_strings or parsed lists in RAM.
    print(f"[STEP 2/3] Processing {n_total:,} rows in chunks of {BATCH_SIZE:,} with {n_workers} workers ...")
    temp_dir = tempfile.mkdtemp(prefix="preprocess_")
    temp_files: list[str] = []
    n_kept = 0
    n_chunks = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    with Pool(n_workers) as pool:
        for chunk_idx in tqdm(range(n_chunks), desc="  Chunks", unit="chunk"):
            start = chunk_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, n_total)

            chunk        = df[start:end]
            chunk_strings = chunk["MatrixStr"].to_list()
            chunk_meta   = chunk.drop("MatrixStr")
            del chunk

            # Parse
            parsed = list(pool.imap(_parse_row, chunk_strings, chunksize=CHUNK_SIZE))
            del chunk_strings

            timestamps = [r[0] for r in parsed]
            matrices   = [r[1] for r in parsed]
            del parsed

            # Filter parse errors + PAUSE in one pass
            chunk_df = (
                chunk_meta.with_columns(
                    pl.Series("Timestamp", timestamps, dtype=pl.Int64),
                    pl.Series("MatrixFlat", matrices,  dtype=pl.List(pl.Int32)),
                )
                .filter(
                    (pl.col("Timestamp") != -1)
                    & pl.col("MatrixFlat").is_not_null()
                    & (pl.col("Task") != "PAUSE")
                )
            )
            del timestamps, matrices, chunk_meta

            # Blob detection
            flat_data = chunk_df["MatrixFlat"].to_list()
            chunk_df  = chunk_df.drop("MatrixFlat")

            blob_results = list(pool.imap(process_blob, flat_data, chunksize=CHUNK_SIZE))
            del flat_data

            chunk_df = (
                chunk_df.with_columns(
                    pl.Series("BlobImgFlattened", blob_results, dtype=pl.List(pl.Int32))
                )
                .filter(pl.col("BlobImgFlattened").is_not_null())
            )
            del blob_results

            if len(chunk_df) > 0:
                temp_path = os.path.join(temp_dir, f"chunk_{chunk_idx:05d}.parquet")
                chunk_df.write_parquet(temp_path)
                temp_files.append(temp_path)
                n_kept += len(chunk_df)

            del chunk_df

    del df
    print(f"[STEP 2/3] Done. {n_kept:,} samples after all filters.\n")

    # Step 3: Concat chunks + save
    print(f"[STEP 3/3] Saving to: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_df = pl.concat([pl.read_parquet(f) for f in temp_files])
    final_df.write_parquet(output_path)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[STEP 3/3] Done. {len(final_df):,} samples saved.")

    return final_df
