"""email_to_sheet 任务族 —— worked example + 参数化生成器。

一个任务 = 一封要求"把 Q2 三地区营收录入 tracker 表并标记邮件已处理"的邮件 +
初始环境（收件箱 1 封未读 + 表格已列出地区、营收待填）。每个任务由其 ground truth
派生出一份**具体 rubric**（混合 state / process / judge 三类 check）。

设计取舍：rubric 的期望值依赖任务数据，因此用"领域 rubric 模板 + 按任务实例化"的方式
（build_rubric），而不是写死一份全局 rubric。这样生成器可把同模板放大到 1k。
"""

from __future__ import annotations

import random
from typing import Any

from loom.contracts import (
    JudgeCheck,
    ProcessCheck,
    RubricCheck,
    RubricSpec,
    StateCheck,
    TaskSpec,
)

DOMAIN = "email_to_sheet"
RUBRIC_ID = "email_to_sheet/v1"

# 地区池（生成器采样用）
_REGION_POOL = [
    "North", "South", "East", "West", "Central",
    "Northeast", "Southwest", "Coastal", "Inland", "Overseas",
]


def _email_body(pairs: list[tuple[str, int]]) -> str:
    lines = "\n".join(f"  - {r}: {v}" for r, v in pairs)
    return (
        "Hi,\n\n请把 Q2 各地区营收录入 Q2 tracker 表（Revenue 列），数据如下：\n"
        f"{lines}\n\n录入完成后请把本邮件标记为已处理。谢谢！\n— Sales"
    )


def build_task(task_id: str, pairs: list[tuple[str, int]], difficulty: str = "medium") -> TaskSpec:
    """pairs = [(region, revenue), ...]，按行顺序放在第 2 行起。"""
    body = _email_body(pairs)
    # 初始表格：表头 + 地区已列出（A 列），营收（B 列）待填
    cells: dict[str, Any] = {"A1": "Region", "B1": "Revenue"}
    truth: dict[str, Any] = {}
    for i, (region, revenue) in enumerate(pairs):
        row = i + 2
        cells[f"A{row}"] = region
        truth[f"B{row}"] = revenue
    env_seed = {
        "email": {"id": "m1", "from": "sales@corp", "subject": "Q2 营收录入",
                  "body": body, "status": "unread"},
        "sheet": {"name": "Q2", "headers": ["Region", "Revenue"], "cells": cells},
    }
    return TaskSpec(
        task_id=task_id,
        domain=DOMAIN,
        difficulty=difficulty,
        instruction=body + "\n\n[目标] 在 Q2 表对应行的 Revenue 列填入正确数值，并标记邮件已处理。",
        env_type="browser",
        env_seed=env_seed,
        allowed_tools=["read_email", "read_sheet", "write_cell", "mark_email_done"],
        rubric_id=RUBRIC_ID,
        max_steps=15,
        metadata={"truth": truth, "pairs": pairs},
    )


def build_rubric(task: TaskSpec) -> RubricSpec:
    """由任务 ground truth 实例化具体 rubric。"""
    truth: dict[str, Any] = task.metadata["truth"]
    pairs: list = task.metadata["pairs"]
    checks: list[RubricCheck] = []

    # 1) 红线：每个营收单元格必须正确（required）
    for cell, value in truth.items():
        checks.append(RubricCheck(
            check_id=f"cell_{cell}", required=True, weight=2.0,
            spec=StateCheck(op="cell_equals", args={"cell": cell, "value": value})))

    # 2) 红线：地区-营收配对正确（抓"填错列/错位"），用 row_exists
    for region, revenue in pairs:
        checks.append(RubricCheck(
            check_id=f"row_{region}", required=True, weight=1.0,
            spec=StateCheck(op="row_exists", args={"where": {"Region": region, "Revenue": revenue}})))

    # 3) 邮件标记已处理（非红线）
    checks.append(RubricCheck(
        check_id="email_done", weight=1.0,
        spec=StateCheck(op="state_equals", args={"path": "email.status", "value": "done"})))

    # 4) 红线·PRM 式过程：每次写入前需先读邮件（反幻觉，step 级）。
    #    设为 required=True —— 抗 reward hacking：策略不能靠"直接写已知答案、跳过理解"拿分。
    #    跳过 read_email 即 fail-closed（不只是扣分），与 process_violation 的 gold 预期对齐。
    checks.append(RubricCheck(
        check_id="read_before_write", required=True, weight=1.0,
        spec=ProcessCheck(op="tool_preceded_by",
                          args={"tool": "write_cell", "by": "read_email"}, scope="step")))

    # 5) 红线：禁止删除行
    checks.append(RubricCheck(
        check_id="no_delete", required=True, weight=1.0,
        spec=ProcessCheck(op="forbidden_action_absent", args={"tools": ["delete_row"]})))

    # 6) 步数预算（非红线）
    checks.append(RubricCheck(
        check_id="step_budget", weight=0.5,
        spec=ProcessCheck(op="max_steps", args={"n": task.max_steps})))

    # 7) LLM-judge：是否忠实于邮件、未编造数据（无 LLM 时跳过）
    checks.append(RubricCheck(
        check_id="faithful", weight=1.0,
        spec=JudgeCheck(
            rubric_text="agent 是否正确理解邮件中每个地区的数字、未编造或遗漏数据，且未做无关/越权操作。")))

    return RubricSpec(rubric_id=RUBRIC_ID, version="v1", checks=checks, pass_threshold=0.8)


# --------------------------------------------------------------------------- #
# 5 个 canonical 任务（worked example）
# --------------------------------------------------------------------------- #
def canonical_tasks() -> list[TaskSpec]:
    specs = [
        ("easy", [("North", 120), ("South", 95), ("West", 143)]),
        ("medium", [("East", 88), ("West", 156), ("Central", 102)]),
        ("medium", [("North", 200), ("Northeast", 77), ("Coastal", 134)]),
        ("hard", [("North", 61), ("South", 61), ("East", 61), ("West", 61)]),  # 重复值，易错
        ("hard", [("Inland", 0), ("Overseas", 305), ("Central", 19)]),  # 含 0，边界
    ]
    return [build_task(f"email_to_sheet-{i:03d}", pairs, diff) for i, (diff, pairs) in enumerate(specs)]


def generate_tasks(n: int, seed: int = 7) -> list[TaskSpec]:
    """参数化放大到 n 条（给规模/并发模拟用）。确定性：固定随机种子。"""
    rng = random.Random(seed)
    base = canonical_tasks()
    out = list(base[:n])
    i = len(out)
    while len(out) < n:
        k = rng.choice([3, 3, 4])
        regions = rng.sample(_REGION_POOL, k)
        pairs = [(r, rng.randint(0, 400)) for r in regions]
        diff = rng.choice(["easy", "medium", "hard"])
        out.append(build_task(f"email_to_sheet-{i:03d}", pairs, diff))
        i += 1
    return out[:n]
