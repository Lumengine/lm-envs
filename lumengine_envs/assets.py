"""Asset root resolution.

Resolution order:
1. LMENVS_ASSETS environment variable (explicit override);
2. the repo checkout's assets/ next to this package (dev / git-clone flow);
3. %LOCALAPPDATA%/lumengine-envs/assets — the download cache that
   scripts/fetch_assets.py fills for wheel installs (no repo checkout).
"""
import os
from pathlib import Path


def assets_dir() -> Path:
    env = os.environ.get("LMENVS_ASSETS")
    if env:
        return Path(env)
    repo_assets = Path(__file__).resolve().parents[1] / "assets"
    if repo_assets.is_dir():
        return repo_assets
    local = os.environ.get("LOCALAPPDATA", str(Path.home()))
    return Path(local) / "lumengine-envs" / "assets"


ASSETS = assets_dir()
