"""CLI: python -m bed_generator.scenes --out DIR [--n 100] [--seed 42]"""
import argparse
import logging

from .config import SceneConfig
from .pipeline import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n", type=int, default=100, help="number of scenes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(generate(args.out, args.n, args.seed, SceneConfig()))


if __name__ == "__main__":
    main()
