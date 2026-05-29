from __future__ import annotations

import math
import os
import cv2
import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
MATRIX_ROWS = 27
MATRIX_COLS  = 15
RESIZE_DIM   = 8
UPSCALE      = 5
CHUNK_SIZE   = 1000


def _clip_negative(flat: list) -> list:
    # Match notebook delete_negative: floor at 0, do NOT cap at 255
    return np.maximum(np.array(flat, dtype=np.int32), 0).tolist()


def _ellipse_params(scaled: np.ndarray) -> tuple[float, float, float, float]:
    thresh = cv2.adaptiveThreshold(
        scaled, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return math.nan, math.nan, math.nan, math.nan
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return math.nan, math.nan, math.nan, math.nan
    ellipse = cv2.fitEllipse(largest)
    major, minor = float(ellipse[1][0]), float(ellipse[1][1])
    area = math.pi * (major / 2) * (minor / 2)
    return major, minor, area, float(ellipse[2])


def _process_row(flat: list) -> dict | None:
    mat = np.maximum(np.array(flat, dtype=np.int32), 0).reshape(MATRIX_ROWS, MATRIX_COLS)

    rows_nz = np.any(mat > 0, axis=1)
    cols_nz = np.any(mat > 0, axis=0)
    if not rows_nz.any():
        return None

    rmax = int(np.where(rows_nz)[0][-1])
    cmax = int(np.where(cols_nz)[0][-1])
    crop = mat[0:rmax + 1, 0:cmax + 1]
    h, w = crop.shape

    # 8x8 resize: cv2 bilinear on float64 (match notebook)
    resized = cv2.resize(crop.astype(np.float64), (RESIZE_DIM, RESIZE_DIM),
                         interpolation=cv2.INTER_LINEAR)

    # Upscale x5 Lanczos for ellipse fitting (clip to uint8 to avoid wrap-around)
    img_up = Image.fromarray(crop.clip(0, 255).astype(np.uint8), mode="L")
    img_up = img_up.resize((w * UPSCALE, h * UPSCALE), Image.LANCZOS)
    scaled = np.array(img_up, dtype=np.uint8)

    major, minor, area, angle = _ellipse_params(scaled)

    return {
        "BlobCrop":       crop.flatten().tolist(),
        "BlobResized8x8": resized.flatten().tolist(),
        "Major_Axis":     major,
        "Minor_Axis":     minor,
        "Area":           area,
        "Angle":          angle,
    }


def run(
    input_parquet: str,
    cap_output:    str,
    snn_output:    str,
    n_workers:     int = 4,
) -> pl.DataFrame:

    print(f"\n[STEP 1/4] Loading: {input_parquet}")
    df = pl.read_parquet(input_parquet)
    print(f"[STEP 1/4] Done. {len(df):,} samples.\n")

    # Clip only negatives (match notebook delete_negative) → save cap parquet
    print("[STEP 2/4] Clipping negatives in BlobImgFlattened ...")
    flat_data = df["BlobImgFlattened"].to_list()
    with Pool(n_workers) as pool:
        clipped = list(tqdm(
            pool.imap(_clip_negative, flat_data, chunksize=CHUNK_SIZE),
            total=len(flat_data), desc="  Clipping", unit="row",
        ))
    df = df.with_columns(
        pl.Series("BlobImgFlattened", clipped, dtype=pl.List(pl.Int32))
    )
    del clipped, flat_data

    os.makedirs(os.path.dirname(os.path.abspath(cap_output)), exist_ok=True)
    df.write_parquet(cap_output)
    print(f"[STEP 2/4] Done. Saved to: {cap_output}\n")

    # Extract BlobCrop, BlobResized8x8, and ellipse params
    print(f"[STEP 3/4] Extracting BlobCrop, BlobResized{RESIZE_DIM}x{RESIZE_DIM}, ellipse ...")
    flat_data = df["BlobImgFlattened"].to_list()
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_row, flat_data, chunksize=CHUNK_SIZE),
            total=len(flat_data), desc="  Extracting", unit="row",
        ))
    del flat_data

    valid_mask    = [r is not None and not math.isnan(r["Area"]) for r in results]
    df_valid      = df.filter(pl.Series(valid_mask))
    valid_results = [r for r, ok in zip(results, valid_mask) if ok]
    del results

    df_snn = df_valid.with_columns(
        pl.Series("BlobCrop",       [r["BlobCrop"]       for r in valid_results],
                  dtype=pl.List(pl.Int32)),
        pl.Series("BlobResized8x8", [r["BlobResized8x8"] for r in valid_results],
                  dtype=pl.List(pl.Float64)),
        pl.Series("Major_Axis",     [r["Major_Axis"]     for r in valid_results],
                  dtype=pl.Float64),
        pl.Series("Minor_Axis",     [r["Minor_Axis"]     for r in valid_results],
                  dtype=pl.Float64),
        pl.Series("Area",           [r["Area"]           for r in valid_results],
                  dtype=pl.Float64),
        pl.Series("Angle",          [r["Angle"]          for r in valid_results],
                  dtype=pl.Float64),
    )
    del valid_results, df_valid
    print(f"[STEP 3/4] Done. {len(df_snn):,} samples with valid features.\n")

    print(f"[STEP 4/4] Saving to: {snn_output}")
    os.makedirs(os.path.dirname(os.path.abspath(snn_output)), exist_ok=True)
    df_snn.write_parquet(snn_output)
    print(f"[STEP 4/4] Done. {len(df_snn):,} samples saved.")

    print("\nColumns saved:")
    for col in df_snn.columns:
        print(f"  {col}: {df_snn[col].dtype}")

    return df_snn
