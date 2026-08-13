"""Command-line entry point for Deep Author Dataset Preparation V1.1."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.deep_author import DeepAuthorBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = DeepAuthorBuilder(root=args.root.resolve()).run()
    print(result)


if __name__ == "__main__":
    main()
