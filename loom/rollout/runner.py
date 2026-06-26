"""Rollout runner —— 内部数据生成 harness（不是给客户的 harness）。

跑 policy × env 的循环，产出 Trajectory：reset → (act ⇄ step)* → 终态快照。

★fault attribution：本 runner 在跑完后给每条 rollout 判一个 Outcome——
区分"环境坏了"(ENV_FAULT)、"策略错了"(POLICY_ERROR)、"正常完成"(COMPLETED)。
判别规则：
- env.reset/step/get_state 抛 EnvFault → ENV_FAULT（基建噪声，绝不计信号，应重试）；
- policy.act 抛异常（模型输出非法/崩溃） → POLICY_ERROR（合法的策略失败信号，不重试）；
- 其余正常跑到终态 → COMPLETED（含合法的 reward=0 负样本）。
其它未预期异常不在此吞掉，向上抛给 run_job_with_retries 归为 HARNESS_FAULT。
"""

from __future__ import annotations

import time
from typing import Any

from loom.contracts import Outcome, Step, TaskSpec, Trajectory
from loom.envs import Environment, EnvFault
from loom.obs import span
from loom.rollout.hooks import PreActionHook
from loom.rollout.policy import Policy

_DONE_NAMES = {None, "submit", "done", "finish"}


def run_rollout(
    task: TaskSpec,
    env: Environment,
    policy: Policy,
    hooks: list[PreActionHook] | None = None,
    max_steps: int | None = None,
    attempt: int = 0,
    close_env: bool = True,
) -> Trajectory:
    # close_env=False：env 由调用方（如 EnvPool）管理生命周期，runner 不主动 close。
    hooks = hooks or []
    max_steps = max_steps or task.max_steps
    t0 = time.perf_counter()

    steps: list[Step] = []
    status = "max_steps"
    outcome = Outcome.COMPLETED
    fault_detail: str | None = None
    final_state: dict[str, Any] = {}

    rollout_span = span("loom.rollout", **{
        "loom.task_id": task.task_id, "loom.policy": getattr(policy, "name", "?"),
        "loom.attempt": attempt, "loom.domain": task.domain})
    with rollout_span:
        try:
            obs = env.reset(task.env_seed)
            policy.reset(task, env.tools())
        except EnvFault as e:  # 环境起不来 = 基建故障，不是策略的错
            outcome = Outcome.ENV_FAULT
            fault_detail = f"reset: {e}"
            status = "error"
        else:
            for i in range(max_steps):
                try:
                    action = policy.act(obs)
                except Exception as e:  # noqa: BLE001 — 策略产出非法/崩溃 → 合法的策略失败信号
                    outcome = Outcome.POLICY_ERROR
                    fault_detail = f"policy.act: {type(e).__name__}: {e}"
                    status = "error"
                    break

                if action is None or action.get("name") in _DONE_NAMES:
                    status = "completed"
                    break

                before = obs
                thought = action.get("thought")
                norm_action = {"name": action.get("name"), "args": action.get("args", {}) or {}}

                reason = None
                for h in hooks:
                    reason = h(norm_action, task)
                    if reason:
                        break

                if reason:  # 被拦截：记录但不执行
                    steps.append(Step(index=i, observation=before, thought=thought,
                                      action=norm_action, tool_result={"blocked": reason},
                                      error=f"blocked: {reason}"))
                    continue

                try:
                    with span("loom.step", **{"loom.index": i, "loom.tool": norm_action.get("name") or ""}):
                        new_obs = env.step(norm_action)
                except EnvFault as e:  # 步中环境崩了 = 基建故障
                    outcome = Outcome.ENV_FAULT
                    fault_detail = f"step[{i}]: {e}"
                    status = "error"
                    break

                steps.append(Step(index=i, observation=before, thought=thought, action=norm_action,
                                  tool_result={"last_result": new_obs.get("last_result")},
                                  error=new_obs.get("error")))
                obs = new_obs

            # 只有环境没坏才读终态（坏环境读 state 会再次抛 EnvFault）
            if outcome != Outcome.ENV_FAULT:
                try:
                    final_state = env.get_state()
                except EnvFault as e:
                    outcome = Outcome.ENV_FAULT
                    fault_detail = f"get_state: {e}"
                    status = "error"
        finally:
            if close_env:
                try:
                    env.close()
                except Exception:
                    pass

    cost = {
        "steps": len(steps),
        "latency_s": round(time.perf_counter() - t0, 4),
        "tokens": getattr(policy, "usage", {}),
    }
    return Trajectory(
        task_id=task.task_id,
        attempt=attempt,
        policy=policy.name,
        steps=steps,
        final_state=final_state,
        status=status,
        outcome=outcome,
        fault_detail=fault_detail,
        cost=cost,
        trace_id=f"{task.task_id}:{policy.name}:{attempt}",
    )
