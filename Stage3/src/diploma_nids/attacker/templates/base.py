"""Base class for attack templates.

A template generates a list of dict-shaped flow records compatible with the
UNSW-NB15 schema. The runtime then converts these to a DataFrame and feeds
them through the same Preprocessor used at training time, ensuring that the
detector sees agent-generated samples in the same representation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FlowRecord:
    """Subset of UNSW-NB15 fields actually populated by the templates."""

    proto: str = "tcp"
    service: str = "-"
    state: str = "FIN"
    dur: float = 0.0
    sbytes: int = 0
    dbytes: int = 0
    spkts: int = 0
    dpkts: int = 0
    sttl: int = 64
    dttl: int = 64
    sloss: int = 0
    dloss: int = 0
    sload: float = 0.0
    dload: float = 0.0
    swin: int = 0
    dwin: int = 0
    stcpb: int = 0
    dtcpb: int = 0
    smean: int = 0
    dmean: int = 0
    sjit: float = 0.0
    djit: float = 0.0
    sinpkt: float = 0.0
    dinpkt: float = 0.0
    rate: float = 0.0
    tcprtt: float = 0.0
    synack: float = 0.0
    ackdat: float = 0.0
    trans_depth: int = 0
    response_body_len: int = 0
    is_sm_ips_ports: int = 0
    is_ftp_login: int = 0
    ct_state_ttl: int = 0
    ct_flw_http_mthd: int = 0
    ct_ftp_cmd: int = 0
    ct_srv_src: int = 0
    ct_srv_dst: int = 0
    ct_dst_ltm: int = 0
    ct_src_ltm: int = 0
    ct_src_dport_ltm: int = 0
    ct_dst_sport_ltm: int = 0
    ct_dst_src_ltm: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class AttackTemplate(ABC):
    """Parametric template for one class of attack.

    Subclasses must implement ``generate(n, rng)``: produce ``n`` flow records.
    Configuration is loaded from a YAML file at construction time.
    """

    name: str = "base"
    attack_cat: str = "Normal"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.params = config["params"]

    @abstractmethod
    def generate(self, n: int, rng: np.random.Generator) -> list[FlowRecord]:
        ...

    def _sample_uniform(self, key: str, rng: np.random.Generator) -> float:
        spec = self.params[key]
        return float(rng.uniform(spec["min"], spec["max"]))

    def _sample_normal(self, key: str, rng: np.random.Generator) -> float:
        spec = self.params[key]
        return float(rng.normal(spec["mean"], spec["std"]))
