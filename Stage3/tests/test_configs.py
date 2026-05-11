"""Validate that all shipped YAML configs are loadable and pass minimal sanity checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from diploma_nids.utils import load_yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def all_yaml_files():
    return sorted(CONFIGS_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("yaml_path", all_yaml_files(), ids=lambda p: str(p.relative_to(CONFIGS_DIR)))
def test_yaml_loads(yaml_path):
    cfg = load_yaml(yaml_path)
    assert isinstance(cfg, dict), f"{yaml_path} did not parse to a dict"
    assert len(cfg) > 0, f"{yaml_path} parsed to empty"


def test_unsw_nb15_schema_complete():
    cfg = load_yaml(CONFIGS_DIR / "data" / "unsw_nb15.yaml")
    assert cfg["name"] == "unsw_nb15"
    schema = cfg["schema"]
    for key in ("numeric", "categorical", "label_binary", "label_multiclass"):
        assert key in schema, f"missing {key} in unsw_nb15 schema"
    cats = cfg["attack_categories"]
    assert "Normal" in cats and "Generic" in cats and "Worms" in cats


def test_cnn_lstm_marked_proposed():
    cfg = load_yaml(CONFIGS_DIR / "models" / "cnn_lstm.yaml")
    assert cfg.get("proposed") is True
    assert cfg["name"] == "cnn_lstm"
    assert "conv_blocks" in cfg["architecture"]
    assert "lstm" in cfg["architecture"]


def test_attacker_policy_has_all_states():
    cfg = load_yaml(CONFIGS_DIR / "attacker" / "policy.yaml")
    states = {s["name"] for s in cfg["states"]}
    expected = {
        "NORMAL",
        "ATTACK_DDOS", "ATTACK_SCAN", "ATTACK_BRUTE",
        "ATTACK_EXPLOIT", "ATTACK_EXFIL", "ATTACK_INTREC", "ATTACK_WORM",
        "DRIFT_COV", "DRIFT_PRIOR", "DRIFT_CONCEPT",
    }
    assert expected.issubset(states), f"missing states: {expected - states}"


def test_attacker_policy_transitions_sum_to_one():
    cfg = load_yaml(CONFIGS_DIR / "attacker" / "policy.yaml")
    for state in cfg["states"]:
        total = sum(state["transitions"].values())
        assert abs(total - 1.0) < 1e-6, f"state {state['name']} transitions sum to {total}"
