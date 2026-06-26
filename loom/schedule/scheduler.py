"""Scheduler —— 资源感知 / 隔离 / 并发 / 轻量重试（设计当真，代码轻量）。

并发模型：
- 每资源类一把 asyncio.Semaphore（browser_heavy 少并发 / light 多并发），
  避免重环境打爆内存。
- 每个 rollout 独占自己的 env 实例（env_factory 每次新建）→ 零共享可变状态 = 隔离。
- 同步的 rollout/verify 用 asyncio.to_thread 投入线程池，被信号量真实限流。
- 有限次重试（异常时），峰值并发 / 吞吐 / 通过率全程统计。

横向扩展映射（README）：1 rollout = 1 K8s Job/Pod，scheduler = controller，
本地 asyncio 池是单机 stand-in。
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from loom.config import DEFAULT_CONCURRENCY
from loom.contracts import RewardReport, RubricSpec, TaskSpec, Trajectory
from loom.curate import Record
from loom.verify import Verifier

RolloutFn = Callable[[TaskSpec, int], Trajectory]  # (task, attempt) -> Trajectory
Classify = Callable[[TaskSpec], str]
RubricFor = Callable[[TaskSpec], RubricSpec]


def default_classify(prefer_browser: bool = False) -> Classify:
    def _c(task: TaskSpec) -> str:
        if task.env_type == "browser" and prefer_browser:
            return "browser_heavy"
        return "light"
    return _c


@dataclass
class RolloutStat:
    task_id: str
    trace_id: str
    resource_class: str
    status: str
    passed: bool
    reward: float
    attempts: int
    duration_s: float


@dataclass
class ScheduleResult:
    records: list[Record] = field(default_factory=list)
    stats: list[RolloutStat] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


async def _aschedule(
    tasks: list[TaskSpec],
    rollout_fn: RolloutFn,
    verifier: Verifier,
    rubric_for: RubricFor,
    classify: Classify,
    concurrency: dict[str, int],
    max_attempts: int,
) -> ScheduleResult:
    sems = {cls: asyncio.Semaphore(cap) for cls, cap in concurrency.items()}
    active: Counter = Counter()
    peak: Counter = Counter()
    res = ScheduleResult()
    lock = asyncio.Lock()

    async def worker(task: TaskSpec) -> None:
        cls = classify(task)
        sem = sems.setdefault(cls, asyncio.Semaphore(8))
        async with sem:
            async with lock:
                active[cls] += 1
                peak[cls] = max(peak[cls], active[cls])
            t0 = time.perf_counter()
            traj: Trajectory | None = None
            attempts = 0
            try:
                for attempt in range(max_attempts):
                    attempts = attempt + 1
                    try:
                        traj = await asyncio.to_thread(rollout_fn, task, attempt)
                        break
                    except Exception:
                        if attempt == max_attempts - 1:
                            traj = Trajectory(task_id=task.task_id, attempt=attempt, policy="error",
                                              status="error", trace_id=f"{task.task_id}:error:{attempt}")
                rep: RewardReport = await asyncio.to_thread(
                    verifier.verify, task, rubric_for(task), traj)
            finally:
                async with lock:
                    active[cls] -= 1
            dur = time.perf_counter() - t0
            async with lock:
                res.records.append((task, traj, rep))
                res.stats.append(RolloutStat(
                    task_id=task.task_id, trace_id=traj.trace_id, resource_class=cls,
                    status=traj.status, passed=rep.passed, reward=rep.total_reward,
                    attempts=attempts, duration_s=round(dur, 4)))

    wall0 = time.perf_counter()
    await asyncio.gather(*(worker(t) for t in tasks))
    wall = time.perf_counter() - wall0

    n = len(res.stats)
    passed = sum(1 for s in res.stats if s.passed)
    retried = sum(1 for s in res.stats if s.attempts > 1)
    res.summary = {
        "total": n,
        "wall_clock_s": round(wall, 3),
        "throughput_per_s": round(n / wall, 1) if wall else 0.0,
        "configured_caps": concurrency,
        "peak_concurrency": dict(peak),
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "passed": passed,
        "retried": retried,
        "by_resource_class": dict(Counter(s.resource_class for s in res.stats)),
    }
    return res


def run_schedule(
    tasks: list[TaskSpec],
    rollout_fn: RolloutFn,
    rubric_for: RubricFor,
    verifier: Verifier | None = None,
    classify: Classify | None = None,
    concurrency: dict[str, int] | None = None,
    max_attempts: int = 2,
) -> ScheduleResult:
    return asyncio.run(_aschedule(
        tasks=tasks,
        rollout_fn=rollout_fn,
        verifier=verifier or Verifier(),
        rubric_for=rubric_for,
        classify=classify or default_classify(),
        concurrency=concurrency or dict(DEFAULT_CONCURRENCY),
        max_attempts=max_attempts,
    ))
