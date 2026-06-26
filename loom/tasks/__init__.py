"""Task & Rubric 定义层 —— 契约 + 壁垒。

- canonical_tasks() / generate_tasks(n)：worked example 与参数化放大。
- rubric_for(task)：由任务实例化具体 rubric。
- load_tasks(path)：从 data/tasks/tasks.jsonl 读取，缺失则回退到 canonical。
- materialize(dir, n)：把任务 + rubric 落成声明式 JSON 文件（交付/透明）。
"""

from __future__ import annotations

import json
from pathlib import Path

from loom.contracts import RubricSpec, TaskSpec
from loom.tasks.builtin import (
    DOMAIN,
    RUBRIC_ID,
    build_rubric,
    build_task,
    canonical_tasks,
    generate_tasks,
)

__all__ = [
    "DOMAIN",
    "RUBRIC_ID",
    "build_task",
    "canonical_tasks",
    "generate_tasks",
    "rubric_for",
    "load_tasks",
    "materialize",
]


def rubric_for(task: TaskSpec) -> RubricSpec:
    if task.domain == DOMAIN:
        return build_rubric(task)
    raise ValueError(f"暂不支持的 domain: {task.domain}")


def load_tasks(path: str | Path | None) -> list[TaskSpec]:
    """从目录下 tasks.jsonl 读取；路径缺失或无文件则回退 canonical_tasks()。"""
    if path is None:
        return canonical_tasks()
    p = Path(path)
    jsonl = p / "tasks.jsonl" if p.is_dir() else p
    if not jsonl.exists():
        return canonical_tasks()
    tasks = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(TaskSpec.model_validate_json(line))
    return tasks


def materialize(out_dir: str | Path, n: int | None = None) -> dict:
    """把任务和实例化 rubric 写成声明式文件，便于交付与人工审阅。"""
    out = Path(out_dir)
    (out / "rubrics").mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(n) if n else canonical_tasks()
    with (out / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")
    for t in tasks:
        rb = rubric_for(t)
        (out / "rubrics" / f"{t.task_id}.json").write_text(
            rb.model_dump_json(indent=2), encoding="utf-8")
    return {"tasks": len(tasks), "dir": str(out)}
