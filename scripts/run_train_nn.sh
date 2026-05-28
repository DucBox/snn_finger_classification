#!/bin/bash
# Train dual-input NN (8x8 blob + ellipse area).
# Task: left/right handedness, thumb + index fingers.
#
# Usage:
#   bash scripts/run_train_nn.sh                          # full model
#   bash scripts/run_train_nn.sh --model-type tiny        # quick dev test
#   bash scripts/run_train_nn.sh --output-dir outputs/exp_01 --epochs 50

set -e

MODEL_TYPE="full"
OUTPUT_DIR="outputs/models"
EPOCHS=100
BATCH_SIZE=16
DEVICE="auto"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-type)  MODEL_TYPE="$2";  shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
    --device)      DEVICE="$2";      shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

docker exec snn_finger python3 train_nn.py \
  --data         outputs/snn_finger_processed_data.parquet \
  --output-dir   "$OUTPUT_DIR" \
  --output-model "model_nn_${MODEL_TYPE}.h5" \
  --model-type   "$MODEL_TYPE" \
  --epochs       "$EPOCHS" \
  --batch-size   "$BATCH_SIZE" \
  --device       "$DEVICE"
