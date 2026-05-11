"""Drift injector: applies covariate / prior / concept shift to flow records.

Operates on the list of FlowRecord produced by templates / FSM at each tick.
This keeps drift fully separated from template parameterization, so that
ground truth (the FSM state) remains uncorrupted.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..utils.io import load_yaml
from .templates.base import FlowRecord


class DriftInjector:
    """Apply one of three drift kinds to a list of records.

    Args:
        config: dict loaded from configs/attacker/drift.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str) -> DriftInjector:
        return cls(load_yaml(path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        records: list[FlowRecord],
        kind: Literal["covariate", "prior", "concept"],
        intensity: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> list[FlowRecord]:
        if not records:
            return records
        rng = rng or np.random.default_rng(0)
        if kind == "covariate":
            return self._covariate(records, intensity, rng)
        if kind == "prior":
            # Prior shift is applied at the agent level (see runtime); this
            # method falls back to identity to keep per-tick processing pure.
            return records
        if kind == "concept":
            return self._concept(records, intensity, rng)
        raise ValueError(f"unknown drift kind: {kind}")

    # ------------------------------------------------------------------
    # Implementations
    # ------------------------------------------------------------------
    def _covariate(
        self,
        records: list[FlowRecord],
        intensity: float,
        rng: np.random.Generator,
    ) -> list[FlowRecord]:
        cov = self.config["covariate"]
        feats = list(cov["features"])
        mean_factor = 1.0 + (float(cov["shift_mean_factor"]) - 1.0) * intensity
        std_factor = 1.0 + (float(cov["shift_std_factor"]) - 1.0) * intensity

        out: list[FlowRecord] = []
        for r in records:
            d = r.to_dict()
            for f in feats:
                if f not in d:
                    continue
                val = float(d[f])
                # Affine shift: scale mean, broaden noise around it
                noise = float(rng.normal(0, abs(val) * 0.05 * (std_factor - 1.0) + 1e-6))
                d[f] = max(0.0, val * mean_factor + noise)
            out.append(FlowRecord(**d))
        return out

    def _concept(
        self,
        records: list[FlowRecord],
        intensity: float,
        rng: np.random.Generator,
    ) -> list[FlowRecord]:
        """Concept drift: gradually mask attack signatures.

        Implementation: reduce values of features that are characteristic
        of a chosen attack template (default: ddos), making the attack
        approach the normal distribution. ``intensity`` linearly scales the
        masking factor in [0, 1].
        """
        conc = self.config["concept"]
        intensity = float(np.clip(intensity, 0.0, 1.0))
        out: list[FlowRecord] = []
        for r in records:
            d = r.to_dict()
            # Soften high-rate / large-burst features
            for feat, scale in (("sload", 0.3), ("dload", 0.3), ("sinpkt", 1.0)):
                val = float(d.get(feat, 0.0))
                d[feat] = val * (1.0 - intensity * (1.0 - scale))
            out.append(FlowRecord(**d))
        return out
