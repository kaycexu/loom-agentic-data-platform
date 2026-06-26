"""看板渲染回归 —— trace 来源内容必须被转义，不能在报告里执行（XSS）。"""

from __future__ import annotations

from loom.contracts import CheckResult, RewardReport, TaskSpec, Trajectory
from loom.trace import RunDir, render_report


def test_report_escapes_malicious_trace(tmp_path):
    payload = "<script>alert('xss')</script>"
    task = TaskSpec(task_id="t-xss", domain="email_to_sheet", instruction="x", rubric_id="r")
    traj = Trajectory(task_id="t-xss", trace_id="tr1", policy=payload, steps=[], final_state={})
    rep = RewardReport(
        task_id="t-xss", trace_id="tr1", total_reward=1.0, passed=True,
        checks=[CheckResult(check_id="c1", kind="state", score=1.0, passed=True,
                            weight=1.0, scope="final", rationale=payload)],
    )
    run = RunDir(str(tmp_path / "run"))
    run.write_records([(task, traj, rep)])
    html = render_report(run).read_text(encoding="utf-8")

    assert payload not in html             # 原始 <script> 不应出现
    assert "&lt;script&gt;" in html        # 已被 HTML 转义
