import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train_nn import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dual-input NN (8x8 blob + area)")
    parser.add_argument("--data",         default="outputs/snn_finger_processed_data.parquet")
    parser.add_argument("--output-model", default="model_nn.h5")
    parser.add_argument("--output-dir",   default="outputs", help="Directory to save the model")
    parser.add_argument("--model-type",   default="full", choices=["tiny", "full"])
    parser.add_argument("--task",         default="thumb_lr",
                        choices=["thumb_lr","index_lr","hand_lr","5fingers","10fingers","thumb_index","thumb_others"])
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--device",       default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--no-area",      action="store_true",
                        help="Train without Area feature (blob-only model)")
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
        device_arg   = args.device,
        use_area     = not args.no_area,
    )
