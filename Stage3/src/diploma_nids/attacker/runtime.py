"""Runtime: turn FSM agent ticks into a stream of preprocessed model inputs.

The runtime can either:
    * generate an offline batch (Pandas DataFrame + ground-truth columns) for
      experiments E4, or
    * stream samples to a callback for the online demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from ..utils.io import load_yaml
from ..utils.logging import get_logger
from .agent import AgentTick, FSMAgent
from .drift_injector import DriftInjector
from .templates.base import FlowRecord

logger = get_logger(__name__)


@dataclass
class StreamSample:
    """One emitted record with attached ground truth."""

    record: FlowRecord
    label: int
    attack_cat: str
    state: str
    drift_kind: str | None
    tick_idx: int


def records_to_dataframe(
    samples: list[StreamSample],
    label_binary_col: str = "label",
    label_multiclass_col: str = "attack_cat",
) -> pd.DataFrame:
    rows = []
    for s in samples:
        d = s.record.to_dict()
        d[label_binary_col] = s.label
        d[label_multiclass_col] = s.attack_cat
        d["fsm_state"] = s.state
        d["drift_kind"] = s.drift_kind or ""
        d["tick_idx"] = s.tick_idx
        rows.append(d)
    return pd.DataFrame(rows)


class AttackerRuntime:
    """Orchestrates FSM agent + drift injector and yields samples."""

    def __init__(
        self,
        policy_path: str,
        drift_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.policy = load_yaml(policy_path)
        if seed is not None:
            self.policy["seed"] = seed
        self.rng = np.random.default_rng(int(self.policy.get("seed", 42)))
        self.agent = FSMAgent(self.policy, rng=self.rng)
        self.drift_injector: DriftInjector | None = (
            DriftInjector.from_yaml(drift_path) if drift_path else None
        )
        self.tick_seconds = float(self.policy.get("tick_interval_seconds", 1.0))

    def stream(self, n_ticks: int | None = None) -> Iterator[StreamSample]:
        n_ticks = n_ticks or int(self.policy.get("duration_seconds", 60) / self.tick_seconds)
        for tick in self.agent.run(n_ticks):
            for s in self._materialize(tick):
                yield s

    def collect(self, n_ticks: int | None = None) -> list[StreamSample]:
        return list(self.stream(n_ticks))

    def collect_dataframe(self, n_ticks: int | None = None) -> pd.DataFrame:
        return records_to_dataframe(self.collect(n_ticks))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _materialize(self, tick: AgentTick) -> list[StreamSample]:
        records = tick.records
        if tick.drift_kind and self.drift_injector is not None:
            intensity = self._intensity_for(tick)
            records = self.drift_injector.apply(records, tick.drift_kind, intensity, self.rng)
        return [
            StreamSample(
                record=r,
                label=1 if tick.is_attack else 0,
                attack_cat=tick.attack_cat,
                state=tick.state,
                drift_kind=tick.drift_kind,
                tick_idx=tick.tick_idx,
            )
            for r in records
        ]

    def _intensity_for(self, tick: AgentTick) -> float:
        if not self.drift_injector:
            return 0.0
        cfg = self.drift_injector.config
        kind_key = {"covariate": "covariate", "concept": "concept", "prior": "prior"}[tick.drift_kind]  # noqa: E501
        ramp = float(cfg.get(kind_key, {}).get("ramp_up_seconds", 60))
        return float(np.clip(tick.tick_idx * self.tick_seconds / max(ramp, 1.0), 0.0, 1.0))


def stream_with_callback(
    runtime: AttackerRuntime,
    callback: Callable[[StreamSample], None],
    n_ticks: int | None = None,
) -> None:
    """Online mode: invoke ``callback`` for each sample."""
    for s in runtime.stream(n_ticks):
        callback(s)
