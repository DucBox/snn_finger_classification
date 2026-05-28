#!/bin/bash
# Train CNN (CapFingerId architecture) on BlobImgFlattened (27x15).
# Task: left/right handedness, thumb + index fingers.
#
# Usage:
#   bash scripts/run_train_cnn.sh
#   bash scripts/run_train_cnn.sh --output-dir outputs/exp_cnn --epochs 50

set -e

OUTPUT_DIR="outputs/models"
EPOCHS=100
BATCH_SIZE=50
DEVICE="auto"

# Override with CLI args if provided
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
    --device)      DEVICE="$2";      shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

docker exec snn_finger python3 train_cnn.py \
  --data        outputs/snn_finger_processed_data.parquet \
  --output-dir  "$OUTPUT_DIR" \
  --output-model model_cnn.h5 \
  --epochs      "$EPOCHS" \
  --batch-size  "$BATCH_SIZE" \
  --device      "$DEVICE"
