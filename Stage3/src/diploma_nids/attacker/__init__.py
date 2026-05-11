"""AI-attacker subpackage: templates + FSM agent + drift injector + runtime."""

from .agent import AgentTick, FSMAgent  # noqa: F401
from .drift_injector import DriftInjector  # noqa: F401
from .runtime import AttackerRuntime, StreamSample, records_to_dataframe, stream_with_callback  # noqa: F401
from .templates import (  # noqa: F401
    TEMPLATE_REGISTRY,
    AttackTemplate,
    BruteForceTemplate,
    DDoSTemplate,
    ExfilTemplate,
    ExploitTemplate,
    FlowRecord,
    InternalReconTemplate,
    NormalTemplate,
    PortScanTemplate,
    WormTemplate,
    build_template,
)

__all__ = [
    "AgentTick",
    "FSMAgent",
    "DriftInjector",
    "AttackerRuntime",
    "StreamSample",
    "records_to_dataframe",
    "stream_with_callback",
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
