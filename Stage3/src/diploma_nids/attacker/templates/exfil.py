"""Data-exfiltration template (UNSW-NB15 'Backdoors' / 'Worms' adjacent)."""
from __future__ import annotations

import numpy as np

from .base import AttackTemplate, FlowRecord


class ExfilTemplate(AttackTemplate):
    name = "exfil"
    attack_cat = "Backdoors"

    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        records: list[FlowRecord] = []
        for _ in range(n):
            dur = float(rng.uniform(self.params["session_duration_seconds"]["min"], self.params["session_duration_seconds"]["max"]))
            bps_out = max(1.0, self._sample_normal("bytes_out_per_second", rng))
            bps_in = max(1.0, self._sample_normal("bytes_in_per_second", rng))
            sbytes = int(bps_out * dur)
            dbytes = int(bps_in * dur)
            records.append(FlowRecord(
                proto=self.params.get("proto", "tcp"),
                service="-" if rng.random() < 0.3 else "http",
                state=self.params.get("state", "CON"),
                dur=dur,
                sbytes=sbytes,
                dbytes=dbytes,
                spkts=max(1, int(dur * 5)),
                dpkts=max(1, int(dur * 1)),
                sload=bps_out,
                dload=bps_in,
                sinpkt=200.0 + float(rng.normal(0, 50)),
                dinpkt=1000.0 + float(rng.normal(0, 100)),
                ct_dst_ltm=int(rng.integers(1, 6)),
                ct_state_ttl=int(rng.integers(0, 3)),
            ))
        return records
