"""Trace 存储 —— 每条 rollout 的可追溯记录（JSONL）。

一个 run 目录布局：
    out/<run>/
      trajectories.jsonl   完整 Trajectory
      reports.jsonl        完整 RewardReport
      trace.jsonl          看板/人读用的精简记录（每 rollout 一行）
      schedule.json        调度并发摘要（scale 时）
      quality.json         验证器可靠性指标（eval 时）
      dataset/             curate 产物（sft.jsonl / rl.jsonl / bundle/ / manifest.json）
      report.html          静态看板预览
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from loom.contracts import RewardReport, TaskSpec, Trajectory


def trace_record(
    task: TaskSpec,
    traj: Trajectory,
    rep: RewardReport,
    resource_class: str | None = None,
    duration_s: float | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "trace_id": traj.trace_id,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "policy": traj.policy,
        "status": traj.status,
        "passed": rep.passed,
        "reward": rep.total_reward,
        "step_rewards": rep.step_rewards,
        "resource_class": resource_class,
        "duration_s": duration_s if duration_s is not None else traj.cost.get("latency_s"),
        "tokens": traj.cost.get("tokens", {}),
        "actions": [s.action.get("name") for s in traj.steps],
        "checks": [
            {"check_id": c.check_id, "kind": c.kind, "required": c.required,
             "passed": c.passed, "skipped": c.skipped, "score": c.score,
             "scope": c.scope, "rationale": c.rationale}
            for c in rep.checks
        ],
    }


class RunDir:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_records(
        self,
        records: list[tuple[TaskSpec, Trajectory, RewardReport]],
        classes: Optional[dict[str, str]] = None,
        durations: Optional[dict[str, float]] = None,
    ) -> None:
        classes = classes or {}
        durations = durations or {}
        with self.path("trajectories.jsonl").open("w", encoding="utf-8") as ft, \
             self.path("reports.jsonl").open("w", encoding="utf-8") as fr, \
             self.path("trace.jsonl").open("w", encoding="utf-8") as fc:
            for task, traj, rep in records:
                ft.write(traj.model_dump_json() + "\n")
                fr.write(rep.model_dump_json() + "\n")
                tr = trace_record(task, traj, rep,
                                  resource_class=classes.get(traj.trace_id),
                                  duration_s=durations.get(traj.trace_id))
                fc.write(json.dumps(tr, ensure_ascii=False) + "\n")

    def write_json(self, name: str, obj: Any) -> None:
        self.path(name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def load_trace(self) -> list[dict[str, Any]]:
        p = self.path("trace.jsonl")
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_json(self, name: str) -> Any:
        p = self.path(name)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
