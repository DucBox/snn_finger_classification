import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocess import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CapFingerId raw .txt data")
    parser.add_argument(
        "--data-path",
        default="../CapFingerId/data/",
        help="Directory containing raw CapMatrix_*.txt files",
    )
    parser.add_argument(
        "--output",
        default="outputs/full_data_set.parquet",
        help="Output path for the processed DataFrame (.parquet)",
    )
    args = parser.parse_args()
    run(data_path=args.data_path, output_path=args.output)
