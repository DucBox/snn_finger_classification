import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_extract import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction: crop + resize blob images")
    parser.add_argument("--input",      default="outputs/full_data_set.parquet")
    parser.add_argument("--cap-output", default="outputs/cap_fingered_processed_data.parquet")
    parser.add_argument("--snn-output", default="outputs/snn_finger_processed_data.parquet")
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    run(
        input_parquet = args.input,
        cap_output    = args.cap_output,
        snn_output    = args.snn_output,
        n_workers     = args.workers,
    )
