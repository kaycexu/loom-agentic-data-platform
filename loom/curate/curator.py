"""Curator —— 把验证过的 rollout 变成可交付数据集。

输入：list[(TaskSpec, Trajectory, RewardReport)]。
逻辑：reward 阈值筛选 → 结构 hash 去重 → 难度/域计数配平统计 → 导出三格式 + manifest。
导出格式（docs/design.md §6.5）：
- sft.jsonl     蒸馏/模仿：{instruction, messages(gold trajectory)}
- rl.jsonl      RL：{task, env_seed, rubric_id, reward, step_rewards, trajectory}
- bundle/       Task+Rubric spec（最有价值，可复跑）：tasks.jsonl + rubrics/*.json
- manifest.json provenance + 计数 + reward 分布（可附 quality_metrics）
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loom.contracts import DatasetManifest, RewardReport, RubricSpec, TaskSpec, Trajectory
from loom.verify import VERIFIER_VERSION

Record = tuple[TaskSpec, Trajectory, RewardReport]
RubricFor = Callable[[TaskSpec], RubricSpec]


def _struct_hash(traj: Trajectory) -> str:
    names = [s.action.get("name") for s in traj.steps]
    payload = json.dumps({"t": traj.task_id, "a": names, "s": traj.final_state},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _reward_hist(rewards: list[float]) -> dict[str, int]:
    buckets = {"[0,0.2)": 0, "[0.2,0.4)": 0, "[0.4,0.6)": 0, "[0.6,0.8)": 0, "[0.8,1.0]": 0}
    for r in rewards:
        if r < 0.2: buckets["[0,0.2)"] += 1
        elif r < 0.4: buckets["[0.2,0.4)"] += 1
        elif r < 0.6: buckets["[0.4,0.6)"] += 1
        elif r < 0.8: buckets["[0.6,0.8)"] += 1
        else: buckets["[0.8,1.0]"] += 1
    return buckets


def _sft_record(task: TaskSpec, traj: Trajectory) -> dict[str, Any]:
    messages = [{"role": "user", "content": task.instruction}]
    for s in traj.steps:
        messages.append({"role": "assistant",
                         "content": json.dumps({"thought": s.thought, "action": s.action}, ensure_ascii=False)})
        messages.append({"role": "tool", "content": json.dumps(s.tool_result, ensure_ascii=False)})
    return {"task_id": task.task_id, "domain": task.domain, "instruction": task.instruction,
            "messages": messages}


def _rl_record(task: TaskSpec, traj: Trajectory, rep: RewardReport) -> dict[str, Any]:
    return {"task_id": task.task_id, "domain": task.domain, "env_type": task.env_type,
            "env_seed": task.env_seed, "rubric_id": task.rubric_id,
            "reward": rep.total_reward, "step_rewards": rep.step_rewards,
            "trajectory": json.loads(traj.model_dump_json())}


def curate(
    records: list[Record],
    rubric_for: RubricFor,
    out_dir: str | Path,
    policy_model: str,
    keep_threshold: float = 0.8,
    keep_only_passed: bool = True,
    quality_metrics: dict[str, Any] | None = None,
    dataset_id: str = "loom-dataset",
) -> DatasetManifest:
    out = Path(out_dir)
    (out / "bundle" / "rubrics").mkdir(parents=True, exist_ok=True)

    all_rewards = [rep.total_reward for _, _, rep in records]
    kept: list[Record] = []
    seen: set[str] = set()
    dropped_dup = dropped_lowq = 0

    for task, traj, rep in records:
        if keep_only_passed and not rep.passed:
            dropped_lowq += 1
            continue
        if rep.total_reward < keep_threshold:
            dropped_lowq += 1
            continue
        key = _struct_hash(traj)
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        kept.append((task, traj, rep))

    # 导出
    with (out / "sft.jsonl").open("w", encoding="utf-8") as f:
        for task, traj, _ in kept:
            f.write(json.dumps(_sft_record(task, traj), ensure_ascii=False) + "\n")
    with (out / "rl.jsonl").open("w", encoding="utf-8") as f:
        for task, traj, rep in kept:
            f.write(json.dumps(_rl_record(task, traj, rep), ensure_ascii=False) + "\n")
    with (out / "bundle" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task, _, _ in kept:
            f.write(task.model_dump_json() + "\n")
    for task, _, _ in kept:
        (out / "bundle" / "rubrics" / f"{task.task_id}.json").write_text(
            rubric_for(task).model_dump_json(indent=2), encoding="utf-8")

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        policy_model=policy_model,
        verifier_versions={"verifier": VERIFIER_VERSION},
        counts={
            "total_in": len(records),
            "kept": len(kept),
            "dropped_low_quality": dropped_lowq,
            "dropped_duplicate": dropped_dup,
            "by_domain": dict(Counter(t.domain for t, _, _ in kept)),
            "by_difficulty": dict(Counter(t.difficulty for t, _, _ in kept)),
        },
        reward_distribution=_reward_hist(all_rewards),
        quality_metrics=quality_metrics or {},
        provenance={"formats": ["sft.jsonl", "rl.jsonl", "bundle/"],
                    "keep_threshold": keep_threshold, "keep_only_passed": keep_only_passed},
    )
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
