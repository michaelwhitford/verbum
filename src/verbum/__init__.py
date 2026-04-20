"""verbum — distilling the lambda compiler from LLMs into a portable tensor artifact.

See AGENTS.md for the project's identity and operating philosophy.
See mementum/knowledge/explore/VERBUM.md for the research program.
"""

from verbum._protocol import (
    CompletionResult,
    HealthStatus,
    ServerProps,
    StreamEvent,
    Timings,
)
from verbum.client import AsyncClient, Client
from verbum.experiment import (
    CacheInterceptor,
    Computation,
    Context,
    Graph,
    Interceptor,
    LogInterceptor,
    ProvenanceInterceptor,
    ResourceInterceptor,
    default_interceptors,
)
from verbum.experiment import run as run_experiment
from verbum.probes import (
    Gate,
    Probe,
    ProbeSet,
    ResolvedProbe,
    gate_hash,
    load_gate,
    load_probe_set,
    probe_set_hash,
    resolve_probes,
)
from verbum.results import (
    ProbeRecord,
    Run,
    RunMeta,
    RunWriter,
    SamplingConfig,
    collect_provenance,
    content_hash,
    load_run,
)
from verbum.runner import RunSummary, fire_probe, run_probe_set

__version__ = "0.0.0"

__all__ = [
    "AsyncClient",
    "CacheInterceptor",
    "Client",
    "CompletionResult",
    "Computation",
    "Context",
    "Gate",
    "Graph",
    "HealthStatus",
    "Interceptor",
    "LogInterceptor",
    "Probe",
    "ProbeRecord",
    "ProbeSet",
    "ProvenanceInterceptor",
    "ResolvedProbe",
    "ResourceInterceptor",
    "Run",
    "RunMeta",
    "RunSummary",
    "RunWriter",
    "SamplingConfig",
    "ServerProps",
    "StreamEvent",
    "Timings",
    "__version__",
    "collect_provenance",
    "content_hash",
    "default_interceptors",
    "fire_probe",
    "gate_hash",
    "load_gate",
    "load_probe_set",
    "load_run",
    "probe_set_hash",
    "resolve_probes",
    "run_experiment",
    "run_probe_set",
]
