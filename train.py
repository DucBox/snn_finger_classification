import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train import run, TASK_CONFIGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NN on finger capacitive images")
    parser.add_argument("--data",       default="outputs/snn_finger_processed_data.parquet")
    parser.add_argument("--task",       default="5fingers", choices=list(TASK_CONFIGS))
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default="auto", help="auto | cpu | gpu")
    args = parser.parse_args()

    run(
        data_path  = args.data,
        task       = args.task,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        device_arg = args.device,
    )
