import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train_snn import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN with keras_spiking")
    parser.add_argument("--data",         default="outputs/snn_finger_processed_data.parquet")
    parser.add_argument("--output-model", default="model_snn.h5")
    parser.add_argument("--output-dir",   default="outputs", help="Directory to save the model")
    parser.add_argument("--model-type",   default="full", choices=["tiny", "full"])
    parser.add_argument("--task",         default="thumb_lr",
                        choices=["thumb_lr","index_lr","hand_lr","5fingers","10fingers","thumb_index","thumb_others"])
    parser.add_argument("--epochs",       type=int,   default=70)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--n-steps",      type=int,   default=5)
    parser.add_argument("--dt",           type=float, default=0.3)
    parser.add_argument("--device",       default="auto", choices=["auto", "cpu", "gpu"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_model = os.path.join(args.output_dir, os.path.basename(args.output_model))

    run(
        data_path    = args.data,
        output_model = output_model,
        model_type   = args.model_type,
        task         = args.task,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        n_steps      = args.n_steps,
        dt           = args.dt,
        device_arg   = args.device,
    )
