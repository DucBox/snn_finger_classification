# Finger Identification with Capacitive Images using SNN

Reproduce the CapFingerId paper (Le et al., IUI 2019) and extend it with a Spiking Neural Network
using `keras_spiking`. Supports 7 classification tasks across thumb/index/all fingers with
configurable output targets (handedness, finger type, or 10-class combined).

---

## Requirements

- Docker

No local Python environment needed. All code runs inside the container.

---

## Project Structure

```
snn_finger/
├── data/                       # Raw capacitive sensor files (P1–P20, 199 .txt files)
├── notebooks/                  # Original Colab notebooks (reference)
│   ├── Data_Preperation.ipynb
│   ├── Train_CNN.ipynb
│   ├── Train_Optimize_NN.ipynb
│   └── Optimal_SNN.ipynb
├── scripts/                    # Shell scripts — one per pipeline step
│   ├── run_feature_extract.sh
│   ├── run_train_cnn.sh
│   ├── run_train_nn.sh         # --task, --model-type, --epochs, --output-dir
│   ├── run_train_snn.sh        # --task, --model-type, --n-steps, --dt, --epochs, --output-dir
│   └── run_power_estimate.sh   # --model, --n-steps, --dt
├── src/                        # Core modules
│   ├── preprocess.py           # Raw .txt → parquet (blob detection)
│   ├── feature_extract.py      # BlobCrop, BlobResized8x8 (cv2), ellipse (cv2.fitEllipse)
│   ├── train_cnn.py            # CapFingerId CNN architecture
│   ├── train_nn.py             # Dual-input NN (tiny / full, 7 tasks)
│   ├── train_snn.py            # Spiking NN via keras_spiking (tiny / full, 7 tasks)
│   ├── power_estimate.py       # Energy estimation via keras_spiking.ModelEnergy
│   └── utils.py                # Seeds, participant split, task/label logic
├── outputs/                    # Generated data and models
│   ├── cap_fingered_processed_data.parquet
│   └── snn_finger_processed_data.parquet
├── preprocess.py               # Entry point: raw data → parquet
├── feature_extract.py          # Entry point: parquet → features + ellipse
├── train_cnn.py                # Entry point: train CNN
├── train_nn.py                 # Entry point: train NN
├── train_snn.py                # Entry point: train SNN
├── power_estimate.py           # Entry point: estimate energy
├── Dockerfile
└── requirements.txt
```

---

## Setup

### 1. Build Docker image

```bash
cd snn_finger
docker build -t snn_finger .
```

### 2. Start container

Raw data is already in the `data/` folder. Mount only the project directory:

```bash
docker run -d --name snn_finger \
  -v $(pwd):/app \
  snn_finger tail -f /dev/null
```

---

## Scripts Reference

All scripts in `scripts/` wrap `docker exec snn_finger python3 ...` calls with sensible defaults.

| Script | Key params | Default |
| --- | --- | --- |
| `run_feature_extract.sh` | `--input`, `--cap-output`, `--snn-output`, `--workers` | — |
| `run_train_cnn.sh` | `--output-dir`, `--epochs`, `--batch-size`, `--device` | epochs=100 |
| `run_train_nn.sh` | `--task`, `--model-type`, `--output-dir`, `--epochs`, `--batch-size`, `--device`, `--no-area` | task=thumb\_lr, epochs=100 |
| `run_train_snn.sh` | `--task`, `--model-type`, `--n-steps`, `--dt`, `--output-dir`, `--epochs`, `--device`, `--no-area` | task=thumb\_lr, epochs=70 |
| `run_power_estimate.sh` | `--model` *(required)*, `--n-steps`, `--dt` | n-steps=5, dt=0.3 |

---

## Pipeline

### Step 1 — Preprocess raw data

Parses `.txt` sensor files, runs DFS blob detection, filters single-blob frames, saves
`BlobImgFlattened` (27×15 padded blob) to parquet.

```bash
docker exec snn_finger python3 preprocess.py \
  --data-path /data \
  --output    outputs/full_data_set.parquet
```

Output: `outputs/full_data_set.parquet` — 455k rows, 6 columns.

---

### Step 2 — Feature extraction

Extracts `BlobCrop`, `BlobResized8x8` (8×8 via cv2 bilinear), and ellipse params
(`Area`, `Major_Axis`, `Minor_Axis`, `Angle`) via `cv2.fitEllipse` on Lanczos×5 upscaled blob.

```bash
bash scripts/run_feature_extract.sh
```

Or manually:

```bash
docker exec snn_finger python3 feature_extract.py \
  --input      outputs/cap_fingered_processed_data.parquet \
  --cap-output outputs/cap_fingered_processed_data.parquet \
  --snn-output outputs/snn_finger_processed_data.parquet \
  --workers    4
```

Output: `outputs/snn_finger_processed_data.parquet` — 12 columns:

| Column | Type | Description |
|---|---|---|
| `Participant` | String | P1 – P20 |
| `Handedness` | String | left / right |
| `Finger` | String | thumb / index / middle / ring / little |
| `Task` | String | TAP / DRAG / SCROLL |
| `Timestamp` | Int64 | Global Unix ms |
| `BlobImgFlattened` | List(Int32) | 27×15 padded blob (405 values) |
| `BlobCrop` | List(Int32) | Raw crop from sensor matrix |
| `BlobResized8x8` | List(Float64) | 8×8 bilinear resize (cv2), 64 values |
| `Major_Axis` | Float64 | Major axis in upscaled pixels |
| `Minor_Axis` | Float64 | Minor axis in upscaled pixels |
| `Area` | Float64 | π·(major/2)·(minor/2) |
| `Angle` | Float64 | Ellipse angle in degrees |

---

### Step 3 — Train CNN

Input: `BlobImgFlattened` (27×15) reshaped to (27, 15, 1).  
Architecture: Conv64→Conv64→Pool→Conv128→Conv128→Pool→Dense256→Dense1(sigmoid).

```bash
# Default (100 epochs)
bash scripts/run_train_cnn.sh

# Custom
bash scripts/run_train_cnn.sh --output-dir outputs/exp_cnn --epochs 50
```

Or:

```bash
docker exec snn_finger python3 train_cnn.py \
  --output-dir  outputs/models \
  --epochs      100 \
  --batch-size  50 \
  --device      cpu
```

---

### Step 4 — Train NN

Input: `BlobResized8x8` (64-dim) + `Area` (1-dim), MinMaxScaler normalized.  
Architecture: Dense128/Dense128 → Concat → Dense64 → Dense32 → Dense16 → Dense(n_classes, softmax).

#### Training tasks (`--task`)

| `--task` | Fingers used | Output | Classes |
| --- | --- | --- | --- |
| `thumb_lr` *(default)* | thumb | left / right | 2 |
| `index_lr` | index | left / right | 2 |
| `hand_lr` | all | left / right | 2 |
| `5fingers` | all | thumb / index / middle / ring / little | 5 |
| `10fingers` | all | thumb\_left / thumb\_right / … | 10 |
| `thumb_index` | thumb + index | thumb / index | 2 |
| `thumb_others` | all | thumb / others | 2 |

#### Model sizes (`--model-type`)

| `--model-type` | Use case | Architecture |
|---|---|---|
| `tiny` | Dev / smoke test | 32 / 8 → 16 → n |
| `full` | Full training | 128 / 128 → 64 → 32 → 16 → n (+ BatchNorm) |

```bash
# Quick smoke test
bash scripts/run_train_nn.sh --model-type tiny --task thumb_lr --epochs 2

# Reproduce paper Table 1 cases
bash scripts/run_train_nn.sh --task thumb_lr     --output-dir outputs/exp_nn
bash scripts/run_train_nn.sh --task index_lr     --output-dir outputs/exp_nn
bash scripts/run_train_nn.sh --task hand_lr      --output-dir outputs/exp_nn
bash scripts/run_train_nn.sh --task 5fingers     --output-dir outputs/exp_nn
bash scripts/run_train_nn.sh --task 10fingers    --output-dir outputs/exp_nn
bash scripts/run_train_nn.sh --task thumb_index  --output-dir outputs/exp_nn
bash scripts/run_train_nn.sh --task thumb_others --output-dir outputs/exp_nn
```

Or manually:

```bash
docker exec snn_finger python3 train_nn.py \
  --model-type  full \
  --task        thumb_lr \
  --output-dir  outputs/models \
  --epochs      100 \
  --batch-size  16
```

Output model is named `model_nn_{model-type}_{task}.h5`.

---

### Step 5 — Train SNN

Same dual-input architecture as NN, but replaces ReLU with `keras_spiking.SpikingActivation`.
Trained **from scratch** (not converted from NN) using surrogate gradient backprop.

Input is tiled along the time axis: shape `(batch, n_steps, 64)` and `(batch, n_steps, 1)`.

Supports the same `--task` options as the NN (see table above).

| Param | Default | Description |
|---|---|---|
| `--task` | `thumb_lr` | Training task (see table above) |
| `--n-steps` | 5 | Simulation timesteps |
| `--dt` | 0.3 | Timestep duration (s) |
| `--model-type` | `full` | `tiny` or `full` |

```bash
# Quick smoke test
bash scripts/run_train_snn.sh --model-type tiny --task thumb_lr --epochs 2

# Full training
bash scripts/run_train_snn.sh --task thumb_lr  --output-dir outputs/exp_snn
bash scripts/run_train_snn.sh --task hand_lr   --output-dir outputs/exp_snn

# Custom timesteps
bash scripts/run_train_snn.sh --task thumb_lr --n-steps 10 --dt 0.5 --output-dir outputs/exp_snn_dt05
```

Or manually:

```bash
docker exec snn_finger python3 train_snn.py \
  --model-type  full \
  --task        thumb_lr \
  --output-dir  outputs/models \
  --epochs      70 \
  --n-steps     5 \
  --dt          0.3
```

Output model is named `model_snn_{model-type}_{task}.h5`.

---

### Step 6 — Estimate energy consumption

Requires a trained SNN `.h5` model. Reports energy per inference on CPU, GPU, Loihi,
SpiNNaker, ARM.

```bash
bash scripts/run_power_estimate.sh --model outputs/models/model_snn_full.h5
```

Or:

```bash
docker exec snn_finger python3 power_estimate.py \
  --model    outputs/models/model_snn_full.h5 \
  --n-steps  5 \
  --dt       0.3
```

---

## Participant Split

Participants are sorted numerically (P1 → P20):
- **Train**: P1 – P16 (80%)
- **Test**: P17 – P20 (20%)

Split is deterministic and reproducible across runs.

---

## Reproducibility

Seeds are fixed at multiple levels:

| Level | Value |
|---|---|
| `PYTHONHASHSEED` | 42 (Dockerfile ENV) |
| `TF_DETERMINISTIC_OPS` | 1 (Dockerfile ENV) |
| `numpy`, `random`, `tf.random` | 42 (set at runtime) |
