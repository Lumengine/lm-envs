"""Locate the Lumengine engine's deployed `lm.rl` runtime and put it on sys.path.

This (separate) envs repo depends on the engine's `lm.rl` facade, which is built
into `<LUMENGINE_ROOT>/build/<cfg>/python`. Point `LUMENGINE_ROOT` at the Lumengine
repo; if unset, a sibling `Lumengine2` / `Lumengine` next to this repo is tried.

    import _bootstrap; _bootstrap.bootstrap()
    import lm.rl as rl
"""
import os
import sys
from pathlib import Path

# Robot / world assets live in this repo (next to tasks/).
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def bootstrap():
    root = os.environ.get("LUMENGINE_ROOT")
    if not root:
        side = Path(__file__).resolve().parents[2]   # dir containing this repo
        for cand in (side / "Lumengine2", side / "Lumengine"):
            if (cand / "build").exists():
                root = cand
                break
    if not root:
        raise RuntimeError(
            "LumengineEnvs: set LUMENGINE_ROOT to the Lumengine repo (the one with "
            "build/<cfg>/python); no sibling Lumengine2/Lumengine was found.")
    cfg = os.environ.get("LUMENGINE_BUILD_CONFIG", "Release")
    build_dir = Path(root) / "build" / cfg
    python_dir = build_dir / "python"
    if not python_dir.exists():
        raise RuntimeError(
            f"LumengineEnvs: lm.rl runtime not found at {python_dir} "
            f"(build the engine, or fix LUMENGINE_ROOT).")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(build_dir))
    sys.path.insert(0, str(python_dir))
