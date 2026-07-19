#!/usr/bin/env python
"""Repo-checkout shim — the CLI lives in lumengine_envs.cli (`lumotion-play`
once pip-installed). Kept so `python play.py ...` keeps working from a clone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumengine_envs.cli import play_main  # noqa: E402

if __name__ == "__main__":
    play_main()
