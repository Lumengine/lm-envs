"""Tier 0 — registry <-> configs consistency. Engine-free: never imports lm.rl
or a task module (TaskSpec is pure data; load_task is NOT called here)."""
from pathlib import Path

import pytest

from lumengine_envs.config import BaseConfig, apply_dict, load_yaml
from lumengine_envs.registry import REGISTRY

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"

DOMAINS = {"classic", "locomotion", "manipulation", "hands", "aerial"}


def test_registry_keys_match_spec_ids():
    for key, spec in REGISTRY.items():
        assert key == spec.id, f"registry key {key!r} != spec.id {spec.id!r}"


def test_registry_ids_unique_case_insensitive():
    lowered = [k.lower() for k in REGISTRY]
    assert len(lowered) == len(set(lowered))


@pytest.mark.parametrize("spec", REGISTRY.values(), ids=lambda s: s.id)
def test_spec_sanity(spec):
    assert spec.domain in DOMAINS
    assert spec.default_envs > 0
    assert spec.max_epochs > 0
    assert spec.module and spec.cls
    cfg = spec.config_cls()                      # engine-free instantiation
    assert isinstance(cfg, BaseConfig)
    assert cfg.num_envs > 0


@pytest.mark.parametrize("spec", REGISTRY.values(), ids=lambda s: s.id)
def test_ppo_attr_declared_consistently(spec):
    # ppo_attr is resolved by getattr at load_task time; here we only check the
    # declaration shape (a non-empty attribute name or None).
    assert spec.ppo_attr is None or (isinstance(spec.ppo_attr, str) and spec.ppo_attr)


def test_every_yaml_config_matches_a_registry_id():
    for p in sorted(CONFIGS.glob("*.yaml")):
        assert p.stem in REGISTRY, (
            f"configs/{p.name} has no matching registry id "
            f"(ids: {', '.join(sorted(REGISTRY))})")


@pytest.mark.parametrize(
    "yaml_path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_yaml_applies_cleanly_to_its_config(yaml_path):
    """apply_dict raises on unknown keys — this catches dataclass<->yaml drift."""
    spec = REGISTRY[yaml_path.stem]
    cfg = spec.config_cls()
    apply_dict(cfg, load_yaml(yaml_path))        # KeyError = drift
