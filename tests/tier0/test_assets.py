"""Tier 0 — every registry task's default config points at assets that exist.

Engine-free. Assets with status `fetched` in THIRD_PARTY_LICENSES.md (anymal)
are allowed to be absent on a fresh clone: the test then FAILS with a pointer
to scripts/fetch_assets.py unless the file exists — run the fetch script once
per machine. (A skip would hide a broken clone forever; a directed failure is
the honest signal. Use --deselect if you really don't want Anymal.)
"""
from pathlib import Path

import pytest

from lumengine_envs.registry import REGISTRY

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"

# Assets documented as fetched-not-committed (see THIRD_PARTY_LICENSES.md).
FETCHED_HINT = "missing fetched asset — run:  python scripts/fetch_assets.py"


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
        p = ASSETS / rel
        assert p.exists(), f"{spec.id}.{field} -> assets/{rel} missing. {FETCHED_HINT}"
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
        p = ASSETS / rel
        if not p.exists():
            pytest.fail(f"{spec.id}.{field} missing: assets/{rel}. {FETCHED_HINT}")
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict) and data, f"assets/{rel} is empty or not a mapping"
