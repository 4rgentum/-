"""Smoke tests for utils subpackage."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from diploma_nids.utils import load_json, load_yaml, save_json, save_yaml, set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    a1 = random.random()
    b1 = np.random.rand()

    set_seed(42)
    a2 = random.random()
    b2 = np.random.rand()

    assert a1 == a2
    assert b1 == b2


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    save_yaml({"k": 1, "list": [1, 2, 3], "nested": {"a": "б"}}, p)
    assert load_yaml(p) == {"k": 1, "list": [1, 2, 3], "nested": {"a": "б"}}


def test_json_roundtrip(tmp_path):
    p = tmp_path / "data.json"
    save_json({"x": 1.5, "lang": "ру"}, p)
    assert load_json(p) == {"x": 1.5, "lang": "ру"}


def test_load_yaml_missing(tmp_path):
    p = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml(p)


def test_save_json_creates_parents(tmp_path):
    p = tmp_path / "deep" / "subdir" / "out.json"
    save_json({"ok": True}, p)
    assert p.is_file()
    assert json.loads(p.read_text("utf-8"))["ok"] is True
