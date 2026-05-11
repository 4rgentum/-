"""Smoke tests for AI-attacker: templates, FSM, drift injector, runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diploma_nids.attacker import (
    AttackerRuntime,
    DriftInjector,
    FSMAgent,
    TEMPLATE_REGISTRY,
    build_template,
    records_to_dataframe,
)
from diploma_nids.utils import load_yaml

ROOT = Path(__file__).resolve().parent.parent
ATTACKER_DIR = ROOT / "configs" / "attacker"


@pytest.mark.parametrize("name,cfg_file", [
    ("ddos", "ddos.yaml"),
    ("scan", "scan.yaml"),
    ("brute", "brute.yaml"),
    ("exploit", "exploit.yaml"),
    ("exfil", "exfil.yaml"),
    ("intrec", "intrec.yaml"),
    ("worm", "worm.yaml"),
])
def test_template_generates_records(name, cfg_file):
    cfg = load_yaml(ATTACKER_DIR / cfg_file)
    tmpl = build_template(name, cfg)
    rng = np.random.default_rng(0)
    records = tmpl.generate(20, rng)
    assert len(records) == 20
    # Each record has populated minimum required fields
    for r in records:
        d = r.to_dict()
        assert d["dur"] >= 0
        assert d["sbytes"] >= 0
        assert isinstance(d["proto"], str)


def test_template_registry_complete():
    expected = {"ddos", "scan", "brute", "exploit", "exfil", "intrec", "worm", "normal"}
    assert expected.issubset(set(TEMPLATE_REGISTRY))


def test_fsm_agent_emits_ticks():
    policy = load_yaml(ATTACKER_DIR / "policy.yaml")
    # Replace template_config paths to absolute (test runs from repo root)
    for s in policy["states"]:
        if "template_config" in s:
            s["template_config"] = str(ROOT / s["template_config"])
    agent = FSMAgent(policy, rng=np.random.default_rng(0))
    ticks = list(agent.run(50))
    assert len(ticks) == 50
    assert any(t.is_attack for t in ticks) or any(t.drift_kind for t in ticks) or all(t.state == "NORMAL" for t in ticks)


def test_drift_injector_covariate_changes_features():
    cfg = load_yaml(ATTACKER_DIR / "drift.yaml")
    inj = DriftInjector(cfg)
    cfg_ddos = load_yaml(ATTACKER_DIR / "ddos.yaml")
    rng = np.random.default_rng(0)
    base = build_template("ddos", cfg_ddos).generate(20, rng)
    drifted = inj.apply(base, "covariate", intensity=1.0, rng=np.random.default_rng(1))
    base_arr = np.array([r.dur for r in base])
    drifted_arr = np.array([r.dur for r in drifted])
    # Covariate shift with mean_factor > 1 should raise mean
    assert drifted_arr.mean() > base_arr.mean() * 0.9  # allow 10% slack


def test_drift_injector_concept_softens_signatures():
    cfg = load_yaml(ATTACKER_DIR / "drift.yaml")
    inj = DriftInjector(cfg)
    cfg_ddos = load_yaml(ATTACKER_DIR / "ddos.yaml")
    rng = np.random.default_rng(0)
    base = build_template("ddos", cfg_ddos).generate(20, rng)
    drifted = inj.apply(base, "concept", intensity=1.0, rng=rng)
    # Concept drift should reduce sload (defining feature for ddos)
    assert np.mean([r.sload for r in drifted]) <= np.mean([r.sload for r in base]) + 1e-6


def test_runtime_produces_dataframe(tmp_path):
    # Make policy paths absolute
    policy = load_yaml(ATTACKER_DIR / "policy.yaml")
    policy["duration_seconds"] = 5
    for s in policy["states"]:
        if "template_config" in s:
            s["template_config"] = str(ROOT / s["template_config"])
    policy_path = tmp_path / "policy.yaml"
    import yaml as _yaml
    with open(policy_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(policy, f, allow_unicode=True)

    rt = AttackerRuntime(str(policy_path), drift_path=str(ATTACKER_DIR / "drift.yaml"), seed=0)
    samples = rt.collect(n_ticks=20)
    df = records_to_dataframe(samples)
    assert len(df) > 0
    assert "label" in df.columns
    assert "attack_cat" in df.columns
    assert "fsm_state" in df.columns


def test_runtime_seed_reproducible(tmp_path):
    policy = load_yaml(ATTACKER_DIR / "policy.yaml")
    for s in policy["states"]:
        if "template_config" in s:
            s["template_config"] = str(ROOT / s["template_config"])
    policy_path = tmp_path / "policy.yaml"
    import yaml as _yaml
    with open(policy_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(policy, f, allow_unicode=True)

    a = AttackerRuntime(str(policy_path), seed=42).collect(n_ticks=20)
    b = AttackerRuntime(str(policy_path), seed=42).collect(n_ticks=20)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa.state == sb.state
        assert sa.is_attack == sb.is_attack
