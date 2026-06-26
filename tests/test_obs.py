"""OTel 追踪回归 —— off 模式 no-op 安全；启用后 span 正确生成且可嵌套。"""

from __future__ import annotations

import loom.obs.tracing as t


def test_off_mode_is_noop():
    assert t.setup_tracing("off") is False
    assert t.is_enabled() is False
    with t.span("loom.x", a=1) as s:  # 不报错、返回 None
        assert s is None
    assert t.inject_context() == {}  # 未启用 → 空 carrier


def test_spans_emitted_and_nested():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # 直接绑定 provider 的 tracer，避免全局 set_tracer_provider 的单次限制
    t._ENABLED = True
    t._tracer = provider.get_tracer("loom-test")
    try:
        with t.span("loom.schedule"):
            with t.span("loom.rollout", **{"loom.task_id": "tt"}):
                with t.span("loom.step", **{"loom.tool": "write_cell"}):
                    pass
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert {"loom.schedule", "loom.rollout", "loom.step"} <= set(spans)
        # 嵌套：step 的 parent 是 rollout，rollout 的 parent 是 schedule（同一 trace）
        step, rollout, sched = spans["loom.step"], spans["loom.rollout"], spans["loom.schedule"]
        assert step.parent.span_id == rollout.context.span_id
        assert rollout.parent.span_id == sched.context.span_id
        assert step.context.trace_id == sched.context.trace_id
    finally:
        t._ENABLED = False
        t._tracer = None
