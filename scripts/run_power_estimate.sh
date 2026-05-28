#!/bin/bash
# Estimate energy consumption of a trained SNN model.
#
# Usage:
#   bash scripts/run_power_estimate.sh --model outputs/models/model_snn_full.h5
#   bash scripts/run_power_estimate.sh --model outputs/models/model_snn_full.h5 --n-steps 5 --dt 0.3

set -e

MODEL=""
N_STEPS=5
DT=0.3

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)    MODEL="$2";    shift 2 ;;
    --n-steps)  N_STEPS="$2";  shift 2 ;;
    --dt)       DT="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required"
  echo "Usage: bash scripts/run_power_estimate.sh --model <path/to/model.h5>"
  exit 1
fi

docker exec snn_finger python3 power_estimate.py \
  --model    "$MODEL" \
  --n-steps  "$N_STEPS" \
  --dt       "$DT"
