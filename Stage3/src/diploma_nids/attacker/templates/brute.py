"""Brute-force template (UNSW-NB15 'Reconnaissance' / 'Exploits')."""
from __future__ import annotations

import numpy as np

from .base import AttackTemplate, FlowRecord


_PORT_TO_SERVICE = {22: "ssh", 3389: "-", 80: "http", 443: "http", 21: "ftp"}


class BruteForceTemplate(AttackTemplate):
    name = "brute"
    attack_cat = "Reconnaissance"

    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        records: list[FlowRecord] = []
        target_ports = list(self.params.get("target_ports", [22, 3389, 80, 443]))
        success_rate = float(self.params.get("success_rate", 0.02))
        for _ in range(n):
            port = int(rng.choice(target_ports))
            sbytes = max(1, int(self._sample_normal("bytes_per_attempt", rng)))
            dbytes = int(rng.integers(20, 80)) if rng.random() > success_rate else int(rng.integers(200, 1000))
            jit_ms = float(rng.uniform(self.params["inter_attempt_jitter_ms"]["min"], self.params["inter_attempt_jitter_ms"]["max"]))
            dur = jit_ms / 1000.0
            records.append(FlowRecord(
                proto="tcp",
                service=_PORT_TO_SERVICE.get(port, "-"),
                state="CON" if dbytes > 100 else "REQ",
                dur=dur,
                sbytes=sbytes,
                dbytes=dbytes,
                spkts=int(rng.integers(2, 5)),
                dpkts=int(rng.integers(2, 5)),
                sload=sbytes / max(dur, 1e-3),
                dload=dbytes / max(dur, 1e-3),
                sinpkt=jit_ms,
                dinpkt=jit_ms,
                is_ftp_login=1 if port == 21 else 0,
                ct_dst_ltm=int(rng.integers(10, 60)),
                ct_state_ttl=int(rng.integers(0, 5)),
            ))
        return records
