#!/usr/bin/env python3
"""Thin CLI entry point for the FlowPredict TAT pipeline.

Usage:
    python scripts/train_flowpredict_tat_model.py [--n-cases N] [--seed S]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.flowpredict.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n_cases=args.n_cases, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
