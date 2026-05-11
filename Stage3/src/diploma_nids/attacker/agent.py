"""Finite-state-machine agent: planner over {NORMAL, ATTACK(c), DRIFT(k)}.

Each tick produces a batch of flow records (via the active template) and a
ground-truth label tuple ``(binary_label, attack_cat, fsm_state)`` so that
downstream evaluation can compute time-to-detect and per-state recall.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..utils.io import load_yaml
from ..utils.logging import get_logger
from .templates import build_template
from .templates.base import AttackTemplate, FlowRecord

logger = get_logger(__name__)


@dataclass
class AgentTick:
    """One simulation tick of the FSM agent."""

    tick_idx: int
    state: str
    is_attack: bool
    attack_cat: str
    drift_kind: str | None
    records: list[FlowRecord]


class FSMAgent:
    """Finite-state-machine planner.

    Args:
        policy: dict loaded from configs/attacker/policy.yaml.
        rng:    numpy Generator. Pass a freshly seeded generator for
                reproducibility.
    """

    def __init__(self, policy: dict[str, Any], rng: np.random.Generator | None = None) -> None:
        self.policy = policy
        self.rng = rng or np.random.default_rng(int(policy.get("seed", 42)))
        self._states: dict[str, dict[str, Any]] = {s["name"]: s for s in policy["states"]}
        if "NORMAL" not in self._states:
            raise ValueError("policy must define a NORMAL state")
        self._templates: dict[str, AttackTemplate] = {}
        self._load_templates()
        self._current = "NORMAL"
        self._ticks_in_state = 0
        self._tick_idx = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _load_templates(self) -> None:
        from .templates.normal import NormalTemplate

        self._templates["NORMAL"] = NormalTemplate()
        for state in self.policy["states"]:
            tmpl = state.get("template")
            if not tmpl:
                continue
            cfg_path = Path(state["template_config"])
            cfg = load_yaml(cfg_path)
            self._templates[state["name"]] = build_template(tmpl, cfg)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(self) -> AgentTick:
        state = self._states[self._current]
        # Sample number of records this tick
        rec_spec = self.policy["samples_per_tick"]
        n = int(self.rng.integers(rec_spec["min"], rec_spec["max"] + 1))

        # Templates by state
        if self._current == "NORMAL":
            records = self._templates["NORMAL"].generate(n, self.rng)
            attack_cat = "Normal"
            is_attack = False
            drift_kind = None
        elif self._current.startswith("ATTACK_"):
            tmpl = self._templates[self._current]
            records = tmpl.generate(n, self.rng)
            attack_cat = tmpl.attack_cat
            is_attack = True
            drift_kind = None
        elif self._current.startswith("DRIFT_"):
            # DRIFT states generate normal-shaped traffic; the actual drift is
            # applied by DriftInjector at the runtime layer (below).
            records = self._templates["NORMAL"].generate(n, self.rng)
            attack_cat = "Normal"
            is_attack = False
            drift_kind = state.get("drift_kind")
        else:
            raise RuntimeError(f"unknown state: {self._current}")

        tick = AgentTick(
            tick_idx=self._tick_idx,
            state=self._current,
            is_attack=is_attack,
            attack_cat=attack_cat,
            drift_kind=drift_kind,
            records=records,
        )

        self._ticks_in_state += 1
        self._tick_idx += 1
        self._maybe_transition(state)
        return tick

    def _maybe_transition(self, state: dict[str, Any]) -> None:
        # Honor minimum duration (as ticks; tick_interval ~1s)
        min_ticks = int(state["min_duration_seconds"])
        if self._ticks_in_state < min_ticks:
            return
        # Sample next state from transition distribution
        trans: dict[str, float] = state["transitions"]
        names = list(trans.keys())
        probs = np.asarray([trans[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        next_state = str(self.rng.choice(names, p=probs))
        if next_state != self._current:
            logger.info("FSM: %s -> %s", self._current, next_state)
            self._current = next_state
            self._ticks_in_state = 0

    # ------------------------------------------------------------------
    # Iterator interface
    # ------------------------------------------------------------------
    def run(self, n_ticks: int) -> Iterator[AgentTick]:
        for _ in range(n_ticks):
            yield self.step()

    @property
    def current_state(self) -> str:
        return self._current
