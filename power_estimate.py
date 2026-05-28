import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.power_estimate import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate SNN energy consumption")
    parser.add_argument("--model",   required=True, help="Path to trained .h5 SNN model")
    parser.add_argument("--n-steps", type=int,   default=5)
    parser.add_argument("--dt",      type=float, default=0.3)
    args = parser.parse_args()

    run(
        model_path = args.model,
        n_steps    = args.n_steps,
        dt         = args.dt,
    )
