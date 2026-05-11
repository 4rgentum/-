"""Attack templates registry."""
from __future__ import annotations

from .base import AttackTemplate, FlowRecord
from .brute import BruteForceTemplate
from .ddos import DDoSTemplate
from .exfil import ExfilTemplate
from .exploit import ExploitTemplate
from .intrec import InternalReconTemplate
from .normal import NormalTemplate
from .scan import PortScanTemplate
from .worm import WormTemplate

TEMPLATE_REGISTRY: dict[str, type[AttackTemplate]] = {
    "ddos": DDoSTemplate,
    "scan": PortScanTemplate,
    "brute": BruteForceTemplate,
    "exploit": ExploitTemplate,
    "exfil": ExfilTemplate,
    "intrec": InternalReconTemplate,
    "worm": WormTemplate,
    "normal": NormalTemplate,
}


def build_template(name: str, config: dict) -> AttackTemplate:
    if name not in TEMPLATE_REGISTRY:
        raise KeyError(f"unknown template: {name}; available: {sorted(TEMPLATE_REGISTRY)}")
    return TEMPLATE_REGISTRY[name](config)


__all__ = [
    "AttackTemplate",
    "FlowRecord",
    "BruteForceTemplate",
    "DDoSTemplate",
    "ExfilTemplate",
    "ExploitTemplate",
    "InternalReconTemplate",
    "NormalTemplate",
    "PortScanTemplate",
    "WormTemplate",
    "TEMPLATE_REGISTRY",
    "build_template",
]
