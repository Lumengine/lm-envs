"""Repo-checkout shim — asset fetching lives in lumotion_envs.assets
(`lumotion-fetch-assets` once pip-installed). Kept so
`python scripts/fetch_assets.py ...` keeps working from a clone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumotion_envs.assets import main  # noqa: E402

if __name__ == "__main__":
    main()
