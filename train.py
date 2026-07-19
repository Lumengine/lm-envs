#!/usr/bin/env python
"""Repo-checkout shim — the CLI lives in lumengine_envs.cli (`lumotion-train`
once pip-installed). Kept so `python train.py ...` keeps working from a clone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumengine_envs.cli import train_main  # noqa: E402

if __name__ == "__main__":
    train_main()
