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
from loom.contracts import TaskSpec
from loom.curate import Record
from loom.obs import inject_context, span
from loom.schedule.executor import make_executor
from loom.schedule.jobs import Job, JobResult
from loom.schedule.store import RunStore

PolicyFor = Callable[[TaskSpec, int], str]
ClassFor = Callable[[TaskSpec, int], str]
PriorityFor = Callable[[TaskSpec, int], int]


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

    records: list[Record] = [
        (tasks_by_id[r.task_id], r.trajectory, r.report)
        for r in all_results if r.task_id in tasks_by_id
    ]

    n = len(all_results)
    passed = sum(1 for r in all_results if r.report.passed)
    dead = [r for r in all_results if r.status == "dead"]
    summary = {
        "run_id": run_id, "executor": executor, "total": n,
        "ran_now": len(results), "resumed_skipped": len(done),
        "wall_clock_s": round(wall, 3),
        "throughput_per_s": round(len(results) / wall, 1) if wall else 0.0,
        "configured_caps": caps, "peak_concurrency": peak,
        "pass_rate": round(passed / n, 4) if n else 0.0, "passed": passed,
        "retried": sum(1 for r in all_results if r.attempts > 1),
        "dead_letter": len(dead),
        "dead_letter_samples": [r.task_id for r in dead][:10],
        "by_resource_class": dict(Counter(r.resource_class for r in all_results)),
    }
    if store:
        store.close()
    return ScheduleResult(records=records, job_results=all_results, summary=summary)
