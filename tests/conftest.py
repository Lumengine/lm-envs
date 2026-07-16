"""Shared pytest infrastructure for LumengineEnvs.

Skip policy (real skips, not RuntimeErrors):
- `@pytest.mark.engine` tests skip when LUMENGINE_ROOT is unset or the built
  `lm.rl` runtime is missing.
- `@pytest.mark.gpu` tests skip when torch has no CUDA device.
- Tier 0 (tests/tier0) uses neither marker and must run everywhere.
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Make `lumengine_envs` importable regardless of how pytest was invoked.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def engine_python_dir():
    """Path of the engine's deployed python dir, or None if not available."""
    root = os.environ.get("LUMENGINE_ROOT")
    if not root:
        return None
    cfg = os.environ.get("LUMENGINE_BUILD_CONFIG", "Release")
    p = Path(root) / "build" / cfg / "python"
    return p if p.exists() else None


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_engine = pytest.mark.skip(
        reason="engine not available: set LUMENGINE_ROOT to a built Lumengine repo")
    skip_gpu = pytest.mark.skip(reason="CUDA not available (direct-GPU batch required)")
    have_engine = engine_python_dir() is not None
    have_cuda = cuda_available()
    for item in items:
        if "engine" in item.keywords and not have_engine:
            item.add_marker(skip_engine)
        if "gpu" in item.keywords and not have_cuda:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def repo_root():
    return REPO


@pytest.fixture(scope="session")
def assets_dir():
    return REPO / "assets"
