#!/bin/bash
# Train SNN using keras_spiking (trained from scratch, not converted from NN).
# Task: left/right handedness, thumb + index fingers.
#
# Usage:
#   bash scripts/run_train_snn.sh                          # full model
#   bash scripts/run_train_snn.sh --model-type tiny        # quick dev test
#   bash scripts/run_train_snn.sh --output-dir outputs/exp_snn --n-steps 10 --dt 0.5

set -e

MODEL_TYPE="full"
TASK="thumb_lr"
OUTPUT_DIR="outputs/models"
EPOCHS=70
BATCH_SIZE=16
N_STEPS=5
DT=0.3
DEVICE="auto"
NO_AREA=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-type)  MODEL_TYPE="$2";  shift 2 ;;
    --task)        TASK="$2";        shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
    --n-steps)     N_STEPS="$2";     shift 2 ;;
    --dt)          DT="$2";          shift 2 ;;
    --device)      DEVICE="$2";      shift 2 ;;
    --no-area)     NO_AREA="--no-area"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

AREA_SUFFIX=""
[[ -n "$NO_AREA" ]] && AREA_SUFFIX="_noarea"

docker exec snn_finger python3 train_snn.py \
  --data         outputs/snn_finger_processed_data.parquet \
  --output-dir   "$OUTPUT_DIR" \
  --output-model "model_snn_${MODEL_TYPE}_${TASK}${AREA_SUFFIX}.h5" \
  --model-type   "$MODEL_TYPE" \
  --task         "$TASK" \
  --epochs       "$EPOCHS" \
  --batch-size   "$BATCH_SIZE" \
  --n-steps      "$N_STEPS" \
  --dt           "$DT" \
  --device       "$DEVICE" \
  $NO_AREA
