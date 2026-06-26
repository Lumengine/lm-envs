"""Phase 0 / 0.3 guard: the lm.rl package must not import engine-internal modules.

For the eventual `lm.rl` pip wheel to be a mechanical extraction, no module under
`lm/rl/` may import an engine-INTERNAL module — `lm.physx`, `lm.physics`,
`lm.scene`, `lm.rendering` (the ECS/PhysX guts) — nor a private cross-package
symbol. Engine access is allowed only through the documented public surface
(`lm.core`, `lm.app`, `lm.rhi`, `lm.bootstrap`, `lm.viewer`, `lm.lumydra`) and the
`lm.rl_backend` façade (a sibling of `lm/rl/`, not shipped in the wheel).

This is a static AST check over the DEPLOYED `lm/rl/*.py` source — it needs no
engine runtime, GPU or USD, so it runs in CI. Point LUMENGINE_ROOT at the engine
repo (same convention as tasks/_bootstrap.py).

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    python tests/test_rl_boundaries.py
"""
import ast
import os
import sys
from pathlib import Path

# Engine-internal modules lm.rl must never import directly (routed via lm.rl_backend).
FORBIDDEN = ("lm.physx", "lm.physics", "lm.scene", "lm.rendering")

# Strict (0.3b landed): every lm/rl module routes engine-internal access through
# lm.rl_backend, so no file is exempt. If a future module legitimately needs a new
# engine internal, add it to lm.rl_backend's re-exports — do NOT add it here.
PENDING = set()


def _rl_dir():
    """Locate the deployed lm/rl package dir from LUMENGINE_ROOT (no import — the
    check is purely static, so no engine DLLs are loaded)."""
    root = os.environ.get("LUMENGINE_ROOT")
    if not root:
        raise RuntimeError(
            "test_rl_boundaries: LUMENGINE_ROOT is not set (point it at the engine "
            "repo with build/<cfg>/python).")
    cfg = os.environ.get("LUMENGINE_BUILD_CONFIG", "Release")
    rl_dir = Path(root) / "build" / cfg / "python" / "lm" / "rl"
    if not rl_dir.exists():
        raise RuntimeError(f"test_rl_boundaries: deployed lm/rl not found at {rl_dir}")
    return rl_dir


def _imported_modules(py_file):
    """All module names imported by a file (both `import X` and `from X import ...`)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:   # absolute import only
                names.append(node.module)
    return names


def _violations(py_file):
    bad = []
    for mod in _imported_modules(py_file):
        if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN):
            bad.append(mod)
    return bad


def run():
    rl_dir = _rl_dir()
    failures = []
    pending_seen = []
    for py_file in sorted(rl_dir.glob("*.py")):
        bad = _violations(py_file)
        if not bad:
            continue
        if py_file.name in PENDING:
            pending_seen.append((py_file.name, bad))
        else:
            failures.append((py_file.name, bad))

    for name, bad in pending_seen:
        print(f"[test] PENDING (0.3b): {name} still imports {sorted(set(bad))}")
    if failures:
        for name, bad in failures:
            print(f"[test] BOUNDARY VIOLATION: lm/rl/{name} imports {sorted(set(bad))}")
        return 1
    print(f"[test] lm.rl boundary OK ({len(list(rl_dir.glob('*.py')))} modules, "
          f"no forbidden engine-internal imports)")
    return 0


def test_rl_boundaries():
    assert run() == 0


if __name__ == "__main__":
    # NB: this static check returns normally (unlike the sim tests, which hard-exit
    # via destroy_world), so let SystemExit propagate — catching BaseException here
    # would turn a clean sys.exit(0) into a false failure.
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        sys.exit(1)
