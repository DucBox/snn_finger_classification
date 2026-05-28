#!/bin/bash
# Re-run feature extraction: generates BlobResized8x8 (cv2), ellipse params (Area, etc.)
# Input:  outputs/cap_fingered_processed_data.parquet
# Output: outputs/snn_finger_processed_data.parquet (updated in-place)

set -e

docker exec snn_finger python3 feature_extract.py \
  --input      outputs/cap_fingered_processed_data.parquet \
  --cap-output outputs/cap_fingered_processed_data.parquet \
  --snn-output outputs/snn_finger_processed_data.parquet \
  --workers    4
