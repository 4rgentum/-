"""Normal-traffic generator: emits benign-looking flows.

Intended to validate the marginal-correspondence test (Stage 2 §3.4):
distributions of dur, sbytes, dbytes etc. should resemble normal traffic
of UNSW-NB15.
"""
from __future__ import annotations

import numpy as np

from .base import AttackTemplate, FlowRecord


class NormalTemplate(AttackTemplate):
    name = "normal"
    attack_cat = "Normal"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config or {"params": {}})

    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        records: list[FlowRecord] = []
        services = ["http", "https", "dns", "smtp", "-"]
        for _ in range(n):
            service = str(rng.choice(services, p=[0.45, 0.25, 0.15, 0.05, 0.10]))
            dur = float(rng.exponential(0.5))
            sbytes = int(rng.lognormal(6.5, 1.5))
            dbytes = int(rng.lognormal(7.5, 1.5))
            spkts = max(1, int(sbytes / 600))
            dpkts = max(1, int(dbytes / 600))
            records.append(FlowRecord(
                proto="tcp" if rng.random() < 0.85 else "udp",
                service=service,
                state="FIN" if dur > 0.05 else "INT",
                dur=dur,
                sbytes=sbytes,
                dbytes=dbytes,
                spkts=spkts,
                dpkts=dpkts,
                sload=sbytes / max(dur, 1e-3),
                dload=dbytes / max(dur, 1e-3),
                sinpkt=float(rng.exponential(50)),
                dinpkt=float(rng.exponential(50)),
                ct_dst_ltm=int(rng.integers(1, 8)),
                ct_state_ttl=int(rng.integers(0, 3)),
            ))
        return records
