"""可观测性 —— OpenTelemetry 链路追踪（off / console / otlp→Jaeger）。"""

from loom.obs.tracing import (
    extract_context,
    inject_context,
    is_enabled,
    setup_tracing,
    span,
    use_context,
)

__all__ = [
    "setup_tracing",
    "span",
    "is_enabled",
    "inject_context",
    "extract_context",
    "use_context",
]
