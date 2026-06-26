"""Rollout runner —— 内部数据生成 harness（不是给客户的 harness）。

跑 policy × env 的循环，产出 Trajectory：reset → (act ⇄ step)* → 终态快照。
"""

from __future__ import annotations

import time
from typing import Any

from loom.contracts import Step, TaskSpec, Trajectory
from loom.envs import Environment
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
) -> Trajectory:
    hooks = hooks or []
    max_steps = max_steps or task.max_steps
    t0 = time.perf_counter()

    obs = env.reset(task.env_seed)
    policy.reset(task, env.tools())

    steps: list[Step] = []
    status = "max_steps"
    for i in range(max_steps):
        action = policy.act(obs)
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

        new_obs = env.step(norm_action)
        steps.append(Step(index=i, observation=before, thought=thought, action=norm_action,
                          tool_result={"last_result": new_obs.get("last_result")},
                          error=new_obs.get("error")))
        obs = new_obs

    final_state = env.get_state()
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
        cost=cost,
        trace_id=f"{task.task_id}:{policy.name}:{attempt}",
    )
