"""调度编排 —— 把 Tasks 变成 Jobs，交给可插拔 Executor 跑，统一收口。

职责：建 Job（注入 OTel carrier 做跨线程/进程 trace 传播）、断点续跑（按 store 跳过已完成）、
跑 executor、合并续跑的历史结果、产出统一 summary（吞吐/峰值并发/通过率/重试/dead-letter）。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loom.config import DEFAULT_CONCURRENCY
from loom.contracts import Outcome, TaskSpec
from loom.curate import Record
from loom.obs import inject_context, span
from loom.schedule.executor import make_executor
from loom.schedule.jobs import Job, JobResult
from loom.schedule.store import RunStore

PolicyFor = Callable[[TaskSpec, int], str]
ClassFor = Callable[[TaskSpec, int], str]
PriorityFor = Callable[[TaskSpec, int], int]

# 成本/吞吐模型的示意单价（USD/分钟）——browser_heavy 远贵于 light，体现异构资源的成本结构。
# 这是确定性的演示估算，非真实计费；用于让"昂贵类是成本主导"这一事实在 summary 里可见、可复现。
UNIT_COST_PER_MIN = {"light": 0.002, "browser_heavy": 0.05}
_DEFAULT_UNIT_COST_PER_MIN = 0.01  # 未知资源类的兜底单价


def _build_cost_model(all_results: list[JobResult], attempted: int) -> dict[str, Any]:
    """据各 rollout 的累计 duration_s 按资源类估算成本。确定性、可复现。

    所有 attempt（含 quarantined/dead）都真实消耗了资源，故成本按 *全部* all_results 统计。"""
    minutes_by_class: dict[str, float] = {}
    for r in all_results:
        minutes_by_class[r.resource_class] = (
            minutes_by_class.get(r.resource_class, 0.0) + r.duration_s / 60.0
        )
    by_resource_class: dict[str, dict[str, float]] = {}
    est_total = 0.0
    for cls, minutes in minutes_by_class.items():
        unit = UNIT_COST_PER_MIN.get(cls, _DEFAULT_UNIT_COST_PER_MIN)
        cost = minutes * unit
        est_total += cost
        by_resource_class[cls] = {
            "minutes": round(minutes, 4),
            "unit_cost_per_min": unit,
            "est_usd": round(cost, 6),
        }
    return {
        "unit_cost_per_min": dict(UNIT_COST_PER_MIN),
        "est_total_usd": round(est_total, 6),
        "est_per_1k_usd": round(est_total / attempted * 1000, 6) if attempted else 0.0,
        "by_resource_class": by_resource_class,
    }


@dataclass
class ScheduleResult:
    records: list[Record] = field(default_factory=list)  # (task, traj, report) 给下游
    job_results: list[JobResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def schedule_tasks(
    tasks: list[TaskSpec],
    *,
    policy_for: PolicyFor,
    class_for: ClassFor,
    priority_for: Optional[PriorityFor] = None,
    prefer_browser: bool = False,
    use_judge: bool = False,
    executor: str = "async",
    caps: Optional[dict[str, int]] = None,
    max_attempts: int = 2,
    run_id: str = "run",
    store_path: Optional[str] = None,
    resume: bool = False,
) -> ScheduleResult:
    caps = caps or dict(DEFAULT_CONCURRENCY)
    store = RunStore(store_path) if store_path else None
    tasks_by_id = {t.task_id: t for t in tasks}

    done: set[str] = store.completed_task_ids(run_id) if (store and resume) else set()
    exec_ = make_executor(executor)

    with span("loom.schedule", **{"loom.run_id": run_id, "loom.total": len(tasks),
                                  "loom.executor": executor}):
        carrier = inject_context()  # 以 schedule 根 span 作为各 rollout 的父
        jobs: list[Job] = []
        for i, t in enumerate(tasks):
            if t.task_id in done:
                continue
            jobs.append(Job(
                run_id=run_id, task=t, policy_spec=policy_for(t, i),
                prefer_browser=prefer_browser, use_judge=use_judge,
                resource_class=class_for(t, i),
                priority=(priority_for(t, i) if priority_for else 100),
                seq=i, otel_carrier=dict(carrier),
            ))
        t0 = time.perf_counter()
        results, peak = exec_.run(jobs, caps, max_attempts, store=store)
        wall = time.perf_counter() - t0

    # 合并续跑：把跳过的（历史已完成）结果加回来
    prior = [r for r in (store.load_results(run_id) if (store and resume) else []) if r.task_id in done]
    all_results = results + prior

    # 诚实分母：只有 SIGNAL（status=="completed"）的 rollout 才是合法训练信号、才进下游。
    # 基建/我方故障（quarantined / dead）结构上排除——既不进 records，也不进 pass_rate 分母。
    signal = [r for r in all_results if r.status == "completed"]
    records: list[Record] = [
        (tasks_by_id[r.task_id], r.trajectory, r.report)
        for r in signal if r.task_id in tasks_by_id
    ]

    attempted = len(all_results)
    completed = len(signal)  # == signal_count
    passed = sum(1 for r in signal if r.report.passed)
    policy_error = sum(1 for r in signal if r.outcome == Outcome.POLICY_ERROR)
    dead = [r for r in all_results if r.status == "dead"]
    quarantined = [r for r in all_results if r.status == "quarantined"]
    summary = {
        "run_id": run_id, "executor": executor, "total": attempted,
        "ran_now": len(results), "resumed_skipped": len(done),
        "wall_clock_s": round(wall, 3),
        "throughput_per_s": round(len(results) / wall, 1) if wall else 0.0,
        "configured_caps": caps, "peak_concurrency": peak,
        # pass_rate 分母 = 仅 completed（SIGNAL），不含 quarantined/dead——诚实分母。
        "pass_rate": round(passed / completed, 4) if completed else 0.0, "passed": passed,
        "retried": sum(1 for r in all_results if r.attempts > 1),
        "dead_letter": len(dead),
        "dead_letter_samples": [r.task_id for r in dead][:10],
        # 容量视图：仍按全部 all_results 统计（每个 attempt 都占用过资源）。
        "by_resource_class": dict(Counter(r.resource_class for r in all_results)),
        # —— 诚实会计 ——
        "attempted": attempted,              # 全部结果数（含 quarantined/dead）
        "completed": completed,              # SIGNAL 数（status=="completed"）
        "policy_error": policy_error,        # completed 中 outcome==POLICY_ERROR 的子集
        "env_fault_quarantined": len(quarantined),  # env 故障耗尽被隔离的数量
        "signal_count": completed,           # 合法训练信号数（== completed）
        # —— 成本/吞吐模型 ——
        "cost_model": _build_cost_model(all_results, attempted),
    }
    if store:
        store.close()
    return ScheduleResult(records=records, job_results=all_results, summary=summary)
