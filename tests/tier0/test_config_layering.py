"""Tier 0 — the config layering contract: defaults -> yaml -> --set -> flags,
type coercion, and typo protection. Engine-free."""
import pytest

from lumotion_envs.config import (CartpoleConfig, Go2Config, apply_dict, apply_set,
                                   build_config)


def test_precedence_defaults_yaml_set_flags(tmp_path):
    y = tmp_path / "t.yaml"
    y.write_text("num_envs: 128\nseed: 7\n", encoding="utf-8")
    cfg = build_config(CartpoleConfig, yaml_path=str(y),
                       sets=["seed=9"], num_envs=64)
    assert cfg.num_envs == 64          # flag beats yaml
    assert cfg.seed == 9               # --set beats yaml
    assert cfg.force_mag == 400.0      # untouched default survives


def test_none_flags_are_ignored():
    cfg = build_config(CartpoleConfig, num_envs=None, seed=None)
    assert cfg.num_envs == CartpoleConfig.num_envs
    assert cfg.seed == CartpoleConfig.seed


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="unknown config key"):
        apply_dict(CartpoleConfig(), {"nom_envs": 12})


def test_set_requires_key_value():
    with pytest.raises(ValueError, match="key=value"):
        apply_set(CartpoleConfig(), ["numenvs"])


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("Yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
])
def test_bool_coercion(raw, expected):
    cfg = apply_dict(CartpoleConfig(), {"headless": raw})
    assert cfg.headless is expected


def test_numeric_coercion_from_strings():
    cfg = apply_dict(Go2Config(), {"num_envs": "256", "action_scale": "0.75"})
    assert cfg.num_envs == 256 and isinstance(cfg.num_envs, int)
    assert cfg.action_scale == 0.75 and isinstance(cfg.action_scale, float)
