"""Port scan template (UNSW-NB15 class 'Reconnaissance')."""
from __future__ import annotations

import numpy as np

from .base import AttackTemplate, FlowRecord


class PortScanTemplate(AttackTemplate):
    name = "scan"
    attack_cat = "Reconnaissance"

    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        records: list[FlowRecord] = []
        for _ in range(n):
            sbytes = max(1, int(self._sample_normal("bytes_per_flow", rng)))
            dur_ms = max(1.0, self._sample_normal("flow_duration_ms", rng))
            dur = dur_ms / 1000.0
            records.append(FlowRecord(
                proto=self.params.get("proto", "tcp"),
                service="-",
                state="REQ",
                dur=dur,
                sbytes=sbytes,
                dbytes=0,
                spkts=1,
                dpkts=0,
                sttl=int(rng.integers(60, 65)),
                dttl=int(rng.integers(60, 65)),
                sload=sbytes / max(dur, 1e-3),
                dload=0.0,
                sinpkt=dur_ms,
                dinpkt=0.0,
                ct_dst_ltm=int(rng.integers(50, 500)),
                ct_dst_sport_ltm=int(rng.integers(50, 500)),
                ct_state_ttl=int(rng.integers(1, 5)),
            ))
        return records
