"""Gold 集构造 —— 度量验证器本身的"标尺"。

用 MockPolicy 的确定性策略在 canonical 任务上造出"正确 + 3 类错误"轨迹，并**人工口径**
标注 label（should_pass / negative_type / is_redline / expected_failed_checks）。
注意：expected_failed_checks 由策略语义推导（我们知道每种策略破坏了什么），
而非读取 verifier 输出 —— 否则就是循环论证。
"""

from __future__ import annotations

from loom.contracts import GoldSample, TaskSpec
from loom.envs import SheetEmailEnv
from loom.rollout import MockPolicy, run_rollout

_RATIONALE = {
    "correct": "完整正确填写并标记已处理，应通过。",
    "missing_fill": "漏填最后一个地区营收，数据不完整，绝不能通过。",
    "wrong_column": "营收填错位/错值，数据错误，绝不能通过。",
    "process_violation": "终态数值正确，但未读邮件即写入(幻觉风险)且调用了禁用的 delete_row，绝不能通过。",
}


def _expected_failed(task: TaskSpec, strategy: str) -> list[str]:
    """按策略语义推导'预期该挂的 check'（不看 verifier 结果）。"""
    truth = task.metadata["truth"]
    cells = [c for c, _ in sorted(truth.items())]
    pairs = task.metadata["pairs"]
    regions = [r for r, _ in pairs]
    if strategy == "missing_fill":  # 漏最后一格
        return [f"cell_{cells[-1]}", f"row_{regions[-1]}"]
    if strategy == "wrong_column":  # 交换前两格（或单值任务退化为最后一格）
        if len({v for _, v in pairs}) >= 2:
            return [f"cell_{cells[0]}", f"cell_{cells[1]}", f"row_{regions[0]}", f"row_{regions[1]}"]
        return [f"cell_{cells[-1]}", f"row_{regions[-1]}"]
    if strategy == "process_violation":
        return ["no_delete", "read_before_write"]
    return []


def build_gold(tasks: list[TaskSpec]) -> list[GoldSample]:
    samples: list[GoldSample] = []
    for task in tasks:
        for strat in ["correct", "missing_fill", "wrong_column", "process_violation"]:
            traj = run_rollout(task, SheetEmailEnv(), MockPolicy(strat))
            is_neg = strat != "correct"
            samples.append(GoldSample(
                sample_id=f"{task.task_id}:{strat}",
                task_id=task.task_id,
                trajectory=traj,
                should_pass=(strat == "correct"),
                negative_type=("none" if strat == "correct" else strat),  # type: ignore[arg-type]
                is_redline=is_neg,  # 任何错误数据/越权都绝不能被判 pass
                expected_failed_checks=_expected_failed(task, strat),
                human_rationale=_RATIONALE[strat],
            ))
    return samples
