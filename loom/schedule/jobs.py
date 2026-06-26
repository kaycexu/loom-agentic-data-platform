"""Job 描述符 + 模块级执行入口 —— 支撑进程池 / K8s Pod 隔离。

设计要点：Job 与 JobResult 都是可 pickle 的 Pydantic 模型，execute_job 是模块级函数，
因此一个 rollout 可以原样搬到子进程或 K8s Pod 里跑（rollout 隔离从"独立 env 实例"
升级到"独立进程/Pod"）。OTel context 经 otel_carrier 跨进程传播。
"""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field

from loom.contracts import RewardReport, TaskSpec, Trajectory


class Job(BaseModel):
    run_id: str
    task: TaskSpec
    policy_spec: str = "mock:correct"  # "mock:<strat>" | "mock" | "llm"
    prefer_browser: bool = False
    resource_class: str = "light"  # light | browser_heavy
    priority: int = 100  # 数值小 = 优先
    seq: int = 0
    use_judge: bool = False  # 是否启用真实 LLM-judge（默认否，省 token）
    otel_carrier: dict[str, str] = Field(default_factory=dict)  # 跨进程 trace 传播


class JobResult(BaseModel):
    run_id: str
    task_id: str
    resource_class: str
    trajectory: Trajectory
    report: RewardReport
    attempts: int = 1
    status: str = "completed"  # completed | dead
    duration_s: float = 0.0
    error: Optional[str] = None


def _policy_from_spec(spec: str):
    from loom.rollout import LLMPolicy, MockPolicy

    if spec == "llm":
        return LLMPolicy()
    if spec.startswith("mock:"):
        return MockPolicy(spec.split(":", 1)[1])
    if spec == "mock":
        return MockPolicy("correct")
    raise ValueError(f"未知 policy_spec: {spec}")


def execute_job(job: Job) -> JobResult:
    """单次执行：rollout + verify。模块级、可 pickle，进程池/Pod 直接调用。"""
    from loom.envs import make_env
    from loom.obs import extract_context, use_context
    from loom.rollout import run_rollout
    from loom.tasks import rubric_for
    from loom.verify import Verifier, default_judge

    t0 = time.perf_counter()
    with use_context(extract_context(job.otel_carrier)):
        env = make_env(job.task.env_type, prefer_browser=job.prefer_browser)
        policy = _policy_from_spec(job.policy_spec)
        traj = run_rollout(job.task, env, policy)
        judge = default_judge() if (job.use_judge or job.policy_spec == "llm") else None
        rep = Verifier(judge=judge).verify(job.task, rubric_for(job.task), traj)

    return JobResult(
        run_id=job.run_id, task_id=job.task.task_id, resource_class=job.resource_class,
        trajectory=traj, report=rep, attempts=1, status="completed",
        duration_s=round(time.perf_counter() - t0, 4),
    )


def _backoff(attempt: int, seq: int, base: float = 0.02, cap: float = 1.0) -> float:
    """指数退避 + 确定性 jitter（不用随机，便于复现）。"""
    delay = base * (2 ** (attempt - 1)) + (seq % 5) * 0.003
    return min(delay, cap)


def _dead_result(job: Job, attempts: int, error: str | None) -> JobResult:
    traj = Trajectory(task_id=job.task.task_id, policy="dead", status="error",
                      trace_id=f"{job.task.task_id}:dead:{attempts}")
    rep = RewardReport(task_id=job.task.task_id, trace_id=traj.trace_id,
                       total_reward=0.0, passed=False, checks=[], policy="dead")
    return JobResult(run_id=job.run_id, task_id=job.task.task_id, resource_class=job.resource_class,
                     trajectory=traj, report=rep, attempts=attempts, status="dead", error=error)


def run_job_with_retries(job: Job, max_attempts: int = 2) -> JobResult:
    """重试 + 指数退避；耗尽则进 dead-letter（status='dead'），绝不静默丢弃。
    模块级，可被进程池 pickle 调用。"""
    last_err: str | None = None
    for k in range(1, max_attempts + 1):
        try:
            res = execute_job(job)
            res.attempts = k
            return res
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if k < max_attempts:
                time.sleep(_backoff(k, job.seq))
    return _dead_result(job, max_attempts, last_err)
