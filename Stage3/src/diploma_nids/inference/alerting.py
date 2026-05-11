"""Alert formation: severity, dedup, prioritization."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Alert:
    alert_id: str
    timestamp: float
    severity: str            # info | low | medium | high | critical
    score: float
    decision: int            # 0/1
    attack_cat_pred: str | None
    profile: dict[str, Any] = field(default_factory=dict)


def severity_of(score: float, threshold: float) -> str:
    """Map a continuous anomaly score to a five-level severity ladder."""
    if score < threshold:
        return "info"
    excess = (score - threshold) / max(1.0 - threshold, 1e-6)
    if excess < 0.1:
        return "low"
    if excess < 0.3:
        return "medium"
    if excess < 0.6:
        return "high"
    return "critical"


class AlertFormer:
    """Format detector outputs as alert dicts with simple time-window dedup."""

    def __init__(
        self,
        threshold: float,
        dedup_window_seconds: float = 5.0,
        history_size: int = 1000,
    ) -> None:
        self.threshold = float(threshold)
        self.dedup_window = float(dedup_window_seconds)
        self.history: deque[Alert] = deque(maxlen=history_size)
        self._counter = 0

    def maybe_alert(
        self,
        score: float,
        attack_cat_pred: str | None,
        profile: dict[str, Any],
        ts: float | None = None,
    ) -> Alert | None:
        if score < self.threshold:
            return None
        ts = ts or time.time()
        # Dedup: skip if a recent alert for the same attack_cat exists
        for prev in reversed(self.history):
            if ts - prev.timestamp > self.dedup_window:
                break
            if prev.attack_cat_pred == attack_cat_pred and prev.severity == severity_of(score, self.threshold):
                return None
        self._counter += 1
        alert = Alert(
            alert_id=f"alert-{self._counter:08d}",
            timestamp=ts,
            severity=severity_of(score, self.threshold),
            score=float(score),
            decision=int(score >= self.threshold),
            attack_cat_pred=attack_cat_pred,
            profile=profile,
        )
        self.history.append(alert)
        return alert

    def recent(self, n: int = 50) -> list[Alert]:
        return list(self.history)[-n:]
