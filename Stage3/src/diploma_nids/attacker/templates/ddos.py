"""DDoS SYN-flood template (UNSW-NB15 class 'DoS')."""
from __future__ import annotations

import numpy as np

from .base import AttackTemplate, FlowRecord


class DDoSTemplate(AttackTemplate):
    name = "ddos"
    attack_cat = "DoS"

    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        records: list[FlowRecord] = []
        intensity = self._sample_uniform("intensity_pps", rng)  # not used per-flow but logged
        syn_ratio = float(self.params.get("syn_ratio", 0.9))
        for _ in range(n):
            packet_size = max(1, int(self._sample_normal("packet_size_bytes", rng)))
            sbytes = packet_size
            dbytes = 0 if rng.random() < syn_ratio else int(rng.integers(40, 200))
            dur = float(rng.uniform(0.001, 0.05))
            sinpkt = 1000.0 / max(intensity, 1.0)
            records.append(FlowRecord(
                proto=self.params.get("proto", "tcp"),
                service="-",
                state=self.params.get("state", "REQ"),
                dur=dur,
                sbytes=sbytes,
                dbytes=dbytes,
                spkts=1,
                dpkts=0 if dbytes == 0 else 1,
                sttl=int(self.params.get("ttl_source", 64)),
                dttl=64,
                sload=sbytes / max(dur, 1e-3),
                dload=dbytes / max(dur, 1e-3),
                sinpkt=sinpkt,
                dinpkt=0.0,
                ct_state_ttl=int(rng.integers(2, 7)),
                ct_dst_ltm=int(rng.integers(20, 200)),
                ct_dst_src_ltm=int(rng.integers(20, 200)),
            ))
        return records
