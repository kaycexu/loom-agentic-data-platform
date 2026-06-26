"""Trace / 看板 —— 可追溯记录 + 静态 HTML preview。"""

from loom.trace.report import render_report
from loom.trace.store import RunDir, trace_record

__all__ = ["RunDir", "trace_record", "render_report"]
