"""Tier 0 — every registry task's default config points at assets that exist.

Engine-free. Assets with status `fetched` in THIRD_PARTY_LICENSES.md (anymal)
are allowed to be absent on a machine that cannot regenerate them (no engine =
no converter — a hosted CI runner): there they SKIP. On a machine WITH the
engine, a missing fetched asset FAILS with a pointer to scripts/fetch_assets.py
— a dev box has no excuse, and a skip would hide the broken clone forever.
"""
from pathlib import Path

import pytest

from lumotion_envs.registry import REGISTRY

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"


def _engine_available():
    import os
    root = os.environ.get("LUMENGINE_ROOT")
    if not root:
        return False
    cfg = os.environ.get("LUMENGINE_BUILD_CONFIG", "Release")
    return (Path(root) / "build" / cfg / "python").exists()

# Assets documented as fetched-not-committed (see THIRD_PARTY_LICENSES.md),
# keyed by the path prefix their configs reference.
FETCHED_PREFIXES = ("anymal_converted/",)
FETCHED_HINT = "missing fetched asset — run:  python scripts/fetch_assets.py"


def _require(spec_id, field, rel):
    """Missing asset -> skip (unfetchable here) or fail (dev box), per docstring."""
    if (ASSETS / rel).exists():
        return
    if rel.startswith(FETCHED_PREFIXES) and not _engine_available():
        pytest.skip(f"{spec_id}.{field}: assets/{rel} is a fetched asset and this "
                    f"machine has no engine to convert it (fine on hosted CI)")
    pytest.fail(f"{spec_id}.{field} -> assets/{rel} missing. {FETCHED_HINT}")


def _asset_fields(cfg):
    """(field, relative_path) pairs of asset references on a config object."""
    for field in ("robot", "robot_usd", "rl_yaml", "cabinet", "cabinet_yaml"):
        rel = getattr(cfg, field, "") or ""
        if rel:
            yield field, rel


@pytest.mark.parametrize("spec", REGISTRY.values(), ids=lambda s: s.id)
def test_default_assets_exist(spec):
    cfg = spec.config_cls()
    checked = 0
    for field, rel in _asset_fields(cfg):
        _require(spec.id, field, rel)
        checked += 1
    if spec.id == "Cartpole":
        # Cartpole's robot_usd="" means "the vendored cartpole" — check it directly.
        assert (ASSETS / "cartpole_converted").is_dir()
        checked += 1
    if spec.id == "Ant":
        # AntConfig has no asset fields: ant_task.py hardcodes the MJCF + prep yaml
        # (config-drive it like the others during the task consolidation phase).
        assert (ASSETS / "ant.xml").exists() and (ASSETS / "ant.rl.yaml").exists()
        checked += 1
    assert checked, f"{spec.id}: no asset fields found on {type(cfg).__name__}"


@pytest.mark.parametrize("spec", REGISTRY.values(), ids=lambda s: s.id)
def test_rl_yaml_parses(spec):
    import yaml
    cfg = spec.config_cls()
    for field in ("rl_yaml", "cabinet_yaml"):
        rel = getattr(cfg, field, "") or ""
        if not rel:
            continue
        _require(spec.id, field, rel)
        p = ASSETS / rel
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict) and data, f"assets/{rel} is empty or not a mapping"
