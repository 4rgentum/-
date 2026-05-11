"""Internal reconnaissance / lateral movement template ('Reconnaissance')."""
from __future__ import annotations

import numpy as np

from .base import AttackTemplate, FlowRecord


class InternalReconTemplate(AttackTemplate):
    name = "intrec"
    attack_cat = "Reconnaissance"

    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        records: list[FlowRecord] = []
        for _ in range(n):
            dur_ms = max(1.0, self._sample_normal("flow_duration_ms", rng))
            sbytes = max(1, int(self._sample_normal("bytes_per_flow", rng)))
            records.append(FlowRecord(
                proto=self.params.get("proto", "tcp"),
                service=str(rng.choice(["smb", "dns", "http", "-"])),
                state="CON",
                dur=dur_ms / 1000.0,
                sbytes=sbytes,
                dbytes=int(sbytes * float(rng.uniform(0.5, 1.5))),
                spkts=int(rng.integers(2, 6)),
                dpkts=int(rng.integers(2, 6)),
                sload=sbytes / max(dur_ms / 1000.0, 1e-3),
                dload=sbytes / max(dur_ms / 1000.0, 1e-3),
                sinpkt=dur_ms / 4,
                dinpkt=dur_ms / 4,
                ct_dst_ltm=int(rng.integers(20, 80)),
                ct_dst_src_ltm=int(rng.integers(20, 80)),
                ct_srv_dst=int(rng.integers(5, 40)),
                ct_state_ttl=int(rng.integers(0, 4)),
            ))
        return records
