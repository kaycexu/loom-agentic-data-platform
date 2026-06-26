"""Job 描述符 + 模块级执行入口 —— 支撑进程池 / K8s Pod 隔离，并承载 fault attribution 路由。

设计要点：
- Job 与 JobResult 都是可 pickle 的 Pydantic 模型，execute_job 是模块级函数，
  因此一个 rollout 可以原样搬到子进程或 K8s Pod 里跑。OTel context 经 otel_carrier 跨进程传播。
- ★fault attribution 路由（本文件的核心）：
  execute_job 拿 runner 判出的 Outcome 决定怎么收尾——
    · SIGNAL_OUTCOMES（COMPLETED / POLICY_ERROR）→ 跑验证器、产出真实 reward（合法信号）；
    · 非信号（ENV_FAULT / HARNESS_FAULT）→ 不跑验证器（坏轨迹的 reward 是噪声），给占位 report。
  run_job_with_retries 再据 Outcome 决定重试与最终处置：
    · 只对 RETRYABLE_OUTCOMES 幂等重试；SIGNAL 立即返回，绝不浪费重试在"模型真错了"上；
    · 重试耗尽：ENV_FAULT → quarantine（可追溯、不交付），HARNESS_FAULT → dead-letter。
  结果：reward 只在合法 rollout 上计算，基建噪声在结构上不可能漏进训练信号。
"""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field

from loom.contracts import (
    RETRYABLE_OUTCOMES,
    SIGNAL_OUTCOMES,
    Outcome,
    RewardReport,
    TaskSpec,
    Trajectory,
)


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
    # outcome：本 rollout 的归因结论（fault attribution 的一等公民）。
    outcome: Outcome = Outcome.COMPLETED
    # status：生命周期处置——completed（有合法信号）| quarantined（env 故障耗尽）| dead（harness 故障耗尽）。
    status: str = "completed"
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def is_signal(self) -> bool:
        """是否产出合法训练信号（可进数据集）。基建故障一律 False。"""
        return self.status == "completed" and self.outcome in SIGNAL_OUTCOMES


def _policy_from_spec(spec: str):
    from loom.rollout import LLMPolicy, MockPolicy

    if spec == "llm":
        return LLMPolicy()
    if spec.startswith("mock:"):
        return MockPolicy(spec.split(":", 1)[1])
    if spec == "mock":
        return MockPolicy("correct")
    raise ValueError(f"未知 policy_spec: {spec}")


def _acquire_env(job: Job):
    """browser_heavy 走 warm 池（吞吐杠杆 + 崩溃驱逐点）；light 直接造（造价低，池化无收益）。
    池不可用时优雅回退到 make_env。返回 (env, pool|None)。"""
    if job.prefer_browser:
        try:
            from loom.envs.pool import get_pool

            pool = get_pool(job.task.env_type, prefer_browser=True)
            return pool.acquire(), pool
        except Exception:
            pass
    from loom.envs import make_env

    return make_env(job.task.env_type, prefer_browser=job.prefer_browser), None


def _placeholder_report(job: Job, traj: Trajectory) -> RewardReport:
    """非信号（基建/我方故障）rollout 的占位 report：reward=0、不通过、不参与训练信号。"""
    return RewardReport(
        task_id=job.task.task_id, trace_id=traj.trace_id,
        total_reward=0.0, passed=False, checks=[], policy=traj.policy,
    )


def execute_job(job: Job) -> JobResult:
    """单次执行：rollout + （按 outcome）verify。模块级、可 pickle，进程池/Pod 直接调用。

    未预期异常不在此吞——向上抛给 run_job_with_retries 归为 HARNESS_FAULT。"""
    from loom.obs import extract_context, use_context
    from loom.rollout import run_rollout
    from loom.tasks import rubric_for
    from loom.verify import Verifier, default_judge

    t0 = time.perf_counter()
    with use_context(extract_context(job.otel_carrier)):
        policy = _policy_from_spec(job.policy_spec)  # 先建 policy：缺 key 等配置错在此抛（HARNESS），不牵涉 env 池
        env, pool = _acquire_env(job)
        healthy = False
        try:
            traj = run_rollout(job.task, env, policy, close_env=(pool is None))
            healthy = traj.outcome != Outcome.ENV_FAULT
            if traj.outcome in SIGNAL_OUTCOMES:
                judge = default_judge() if (job.use_judge or job.policy_spec == "llm") else None
                rep = Verifier(judge=judge).verify(job.task, rubric_for(job.task), traj)
            else:  # ENV_FAULT：坏轨迹不喂验证器，避免基建噪声变成 reward
                rep = _placeholder_report(job, traj)
        finally:
            if pool is not None:
                try:
                    pool.release(env, healthy=healthy)  # 不健康（曾 EnvFault）→ 驱逐销毁
                except Exception:
                    pass

    return JobResult(
        run_id=job.run_id, task_id=job.task.task_id, resource_class=job.resource_class,
        trajectory=traj, report=rep, attempts=1, outcome=traj.outcome,
        error=traj.fault_detail,
        duration_s=round(time.perf_counter() - t0, 4),
    )


def _backoff(attempt: int, seq: int, base: float = 0.02, cap: float = 1.0) -> float:
    """指数退避 + 确定性 jitter（不用随机，便于复现）。"""
    delay = base * (2 ** (attempt - 1)) + (seq % 5) * 0.003
    return min(delay, cap)


def _harness_fault_result(job: Job, error: str) -> JobResult:
    """execute_job 抛出未预期异常 → 我方故障（HARNESS_FAULT），可重试。"""
    traj = Trajectory(task_id=job.task.task_id, policy="harness", status="error",
                      outcome=Outcome.HARNESS_FAULT, fault_detail=error,
                      trace_id=f"{job.task.task_id}:harness")
    rep = RewardReport(task_id=job.task.task_id, trace_id=traj.trace_id,
                       total_reward=0.0, passed=False, checks=[], policy="harness")
    return JobResult(run_id=job.run_id, task_id=job.task.task_id, resource_class=job.resource_class,
                     trajectory=traj, report=rep, outcome=Outcome.HARNESS_FAULT, error=error)


def _finalize(res: JobResult) -> JobResult:
    """据 outcome 定最终生命周期处置（status）。"""
    if res.outcome == Outcome.ENV_FAULT:
        res.status = "quarantined"  # 环境故障耗尽：可追溯、排除出数据集、需运维介入
    elif res.outcome == Outcome.HARNESS_FAULT:
        res.status = "dead"  # 我方故障耗尽：dead-letter
    else:
        res.status = "completed"  # COMPLETED / POLICY_ERROR：有合法信号
    return res


def run_job_with_retries(job: Job, max_attempts: int = 2) -> JobResult:
    """按 Outcome 路由重试：只对基建/我方故障幂等重试，绝不重试"模型真错了"；
    耗尽则按故障类型进 quarantine / dead-letter，绝不静默丢弃。模块级，可被进程池 pickle 调用。"""
    res: JobResult | None = None
    for k in range(1, max_attempts + 1):
        try:
            res = execute_job(job)
        except Exception as e:  # noqa: BLE001 — 未预期异常 = 我方故障
            res = _harness_fault_result(job, f"{type(e).__name__}: {e}")
        res.attempts = k
        if res.outcome in RETRYABLE_OUTCOMES and k < max_attempts:
            time.sleep(_backoff(k, job.seq))
            continue
        return _finalize(res)
    return _finalize(res)  # safety（理论不可达）
