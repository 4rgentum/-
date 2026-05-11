"""Unit tests for alert formation: severity ladder and dedup."""
from __future__ import annotations

import time

from diploma_nids.inference import AlertFormer, severity_of


def test_severity_levels_below_threshold():
    assert severity_of(0.4, threshold=0.5) == "info"


def test_severity_ladder_above_threshold():
    t = 0.5
    assert severity_of(0.51, t) == "low"
    assert severity_of(0.7, t) == "medium"
    assert severity_of(0.85, t) == "high"
    assert severity_of(0.99, t) == "critical"


def test_alert_former_emits_above_threshold():
    af = AlertFormer(threshold=0.5, dedup_window_seconds=60)
    a = af.maybe_alert(0.95, attack_cat_pred="DoS", profile={})
    assert a is not None
    assert a.severity == "critical"
    assert a.decision == 1


def test_alert_former_skips_below_threshold():
    af = AlertFormer(threshold=0.5)
    assert af.maybe_alert(0.3, attack_cat_pred="DoS", profile={}) is None


def test_alert_former_dedup_within_window():
    af = AlertFormer(threshold=0.5, dedup_window_seconds=60)
    ts = time.time()
    af.maybe_alert(0.95, "DoS", {}, ts=ts)
    second = af.maybe_alert(0.96, "DoS", {}, ts=ts + 1.0)  # same severity, same cat
    assert second is None


def test_alert_former_no_dedup_after_window():
    af = AlertFormer(threshold=0.5, dedup_window_seconds=2.0)
    ts = time.time()
    af.maybe_alert(0.95, "DoS", {}, ts=ts)
    later = af.maybe_alert(0.96, "DoS", {}, ts=ts + 3.0)
    assert later is not None


def test_recent_returns_at_most_n():
    af = AlertFormer(threshold=0.5, history_size=10)
    ts = time.time()
    for i in range(20):
        af.maybe_alert(0.9 + i * 0.001, attack_cat_pred=f"cat-{i}", profile={}, ts=ts + i * 10)
    assert len(af.recent(5)) == 5
    assert len(af.history) == 10
