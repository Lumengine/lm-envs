"""Engine runtime access — the ONE place that locates the `lm` package.

Two supported situations (plan 003, Phase 3):
- wheel install: `pip install lumotion` put `lm` in site-packages and
  `import lm` just works;
- dev checkout: the engine lives in a Lumengine repo's build tree, pointed
  at by LUMENGINE_ROOT (and optionally LUMENGINE_BUILD_CONFIG).

Every task module calls `ensure_engine()` before importing `lumotion`; nothing
else in this repo may touch sys.path for the engine.
"""
import os
import sys
from pathlib import Path


def ensure_engine() -> None:
    """Make `import lm` work, or raise an ImportError that names the remedy."""
    # Torch and the engine must share one CUDA primary context; the engine
    # reads this at PhysX startup. Opt-out by exporting it to 0 explicitly.
    os.environ.setdefault("LM_PHYSX_SHARE_CUDA_CONTEXT", "1")

    try:
        import lumotion  # noqa: F401  (wheel or already-wired dev path)
        return
    except ImportError:
        pass

    root = os.environ.get("LUMENGINE_ROOT")
    if not root:
        raise ImportError(
            "lumotion-envs: cannot import the engine package 'lm'. Either\n"
            "  - install the Lumotion runtime wheel:  pip install lumotion\n"
            "  - or point LUMENGINE_ROOT at a Lumengine repo with a built "
            "engine (expects <LUMENGINE_ROOT>/build/<cfg>/python)")

    cfg = os.environ.get("LUMENGINE_BUILD_CONFIG", "Release")
    build_dir = Path(root) / "build" / cfg
    python_dir = build_dir / "python"
    if not python_dir.exists():
        raise ImportError(
            f"lumotion-envs: LUMENGINE_ROOT={root} but {python_dir} does not "
            f"exist — build the engine, or fix LUMENGINE_ROOT / "
            f"LUMENGINE_BUILD_CONFIG (currently {cfg!r}).")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(build_dir))
    sys.path.insert(0, str(python_dir))
    import lumotion  # noqa: F401 — surfaces the real loader error if still broken
