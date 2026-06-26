"""OpenTelemetry 链路追踪 —— no-op 安全封装。

三种模式（环境变量 LOOM_OTEL 或 setup_tracing(mode=...)）：
  off      （默认）完全 no-op，零开销，CI/离线友好
  console  打到 stdout（离线即可验证 span 树正确）
  otlp     OTLP/HTTP 导出到 Jaeger（OTEL_EXPORTER_OTLP_ENDPOINT，默认 http://localhost:4318）

span 树（schedule → rollout → step / verify → check）跨线程（AsyncExecutor）与
跨进程（ProcessExecutor，经 inject/extract traceparent）传播。
所有 API 在未启用或 opentelemetry 缺失时都是安全 no-op。
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

_ENABLED = False
_tracer: Any = None


def setup_tracing(mode: Optional[str] = None, service_name: str = "loom") -> bool:
    """配置全局 TracerProvider。返回是否启用。重复调用幂等（按当前 provider）。"""
    global _ENABLED, _tracer
    mode = (mode or os.environ.get("LOOM_OTEL", "off")).lower()
    if mode in ("", "off", "none", "0", "false"):
        _ENABLED = False
        _tracer = None
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if mode == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318").rstrip("/")
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")))
        else:  # console
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("loom")
        _ENABLED = True
        return True
    except Exception:
        _ENABLED = False
        _tracer = None
        return False


def is_enabled() -> bool:
    return _ENABLED


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """开启一个 span（未启用时 no-op）。属性键建议用 loom.* 命名。"""
    if not _ENABLED or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            try:
                s.set_attribute(k, v)
            except Exception:
                pass
        yield s


def inject_context() -> dict[str, str]:
    """把当前 trace context 注入到可序列化 carrier（用于跨进程传播）。"""
    if not _ENABLED:
        return {}
    try:
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        return carrier
    except Exception:
        return {}


def extract_context(carrier: dict[str, str] | None) -> Any:
    """从 carrier 还原 context（跨进程 span 链接）。"""
    if not _ENABLED or not carrier:
        return None
    try:
        from opentelemetry.propagate import extract

        return extract(carrier)
    except Exception:
        return None


@contextlib.contextmanager
def use_context(ctx: Any) -> Iterator[None]:
    """在给定 context 下执行（用于线程/进程内还原父 span）。"""
    if not _ENABLED or ctx is None:
        yield
        return
    try:
        from opentelemetry import context as _ctx

        token = _ctx.attach(ctx)
        try:
            yield
        finally:
            _ctx.detach(token)
    except Exception:
        yield
