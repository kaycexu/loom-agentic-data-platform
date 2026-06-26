"""Verifier 引擎单测 —— 证明多层验证能精准区分对/错轨迹，红线零误收。"""

from __future__ import annotations

import pytest

from loom.contracts import (
    JudgeCheck,
    ProcessCheck,
    RubricCheck,
    RubricSpec,
    StateCheck,
    Step,
    TaskSpec,
    Trajectory,
)
from loom.verify import Verifier

TASK = TaskSpec(
    task_id="t-demo",
    domain="email_to_sheet",
    instruction="把邮件里 North=120 South=95 West=143 录入 Q2 表，并标记邮件已处理",
    rubric_id="r-demo",
    allowed_tools=["read_email", "write_cell", "mark_email_done"],
)

RUBRIC = RubricSpec(
    rubric_id="r-demo",
    pass_threshold=0.8,
    checks=[
        RubricCheck(check_id="row_north", required=True, weight=2,
                    spec=StateCheck(op="cell_equals", args={"cell": "B2", "value": 120})),
        RubricCheck(check_id="row_south", required=True, weight=2,
                    spec=StateCheck(op="cell_equals", args={"cell": "B3", "value": 95})),
        RubricCheck(check_id="row_west", required=True, weight=2,
                    spec=StateCheck(op="cell_equals", args={"cell": "B4", "value": 143})),
        RubricCheck(check_id="email_done", weight=1,
                    spec=StateCheck(op="state_equals", args={"path": "email.status", "value": "done"})),
        RubricCheck(check_id="read_first", weight=1,
                    spec=ProcessCheck(op="tool_preceded_by",
                                      args={"tool": "write_cell", "by": "read_email"}, scope="step")),
        RubricCheck(check_id="no_delete", required=True, weight=1,
                    spec=ProcessCheck(op="forbidden_action_absent", args={"tools": ["delete_row"]})),
        RubricCheck(check_id="step_budget", weight=1,
                    spec=ProcessCheck(op="max_steps", args={"n": 15})),
    ],
)

GOOD_STATE = {
    "sheet": {
        "cells": {"A2": "North", "B2": 120, "A3": "South", "B3": 95, "A4": "West", "B4": 143},
        "rows": [
            {"Region": "North", "Revenue": 120},
            {"Region": "South", "Revenue": 95},
            {"Region": "West", "Revenue": 143},
        ],
    },
    "email": {"status": "done"},
}


def _step(i, name, **args):
    return Step(index=i, action={"name": name, "args": args})


def _traj(steps, state, tid="trace-x", policy="mock"):
    return Trajectory(task_id="t-demo", steps=steps, final_state=state, trace_id=tid, policy=policy)


def verify(traj):
    return Verifier().verify(TASK, RUBRIC, traj)


def test_correct_trajectory_passes():
    steps = [
        _step(0, "read_email"),
        _step(1, "write_cell", cell="B2", value=120),
        _step(2, "write_cell", cell="B3", value=95),
        _step(3, "write_cell", cell="B4", value=143),
        _step(4, "mark_email_done"),
    ]
    rep = verify(_traj(steps, GOOD_STATE))
    assert rep.passed is True
    assert rep.total_reward >= 0.8
    # judge 未配置 LLM → 跳过，不污染分数
    assert any(c.check_id == "faithful" for c in rep.checks) is False  # 本 rubric 无 judge


def test_missing_fill_fails_redline():
    """漏填 West → 红线 required 不过 → 必须 fail（误收测试）。"""
    state = {k: v for k, v in GOOD_STATE.items()}
    state["sheet"] = {"cells": dict(GOOD_STATE["sheet"]["cells"]), "rows": list(GOOD_STATE["sheet"]["rows"])}
    del state["sheet"]["cells"]["B4"]  # West 没填
    steps = [_step(0, "read_email"), _step(1, "write_cell", cell="B2", value=120),
             _step(2, "write_cell", cell="B3", value=95), _step(3, "mark_email_done")]
    rep = verify(_traj(steps, state))
    assert rep.passed is False
    assert any(c.check_id == "row_west" and not c.passed for c in rep.checks)


def test_wrong_column_fails():
    """填错值（West 写成 999）→ state 检查 fail。"""
    state = {"email": {"status": "done"},
             "sheet": {"cells": {"B2": 120, "B3": 95, "B4": 999}, "rows": []}}
    steps = [_step(0, "read_email"), _step(1, "write_cell", cell="B4", value=999)]
    rep = verify(_traj(steps, state))
    assert rep.passed is False
    assert any(c.check_id == "row_west" and not c.passed for c in rep.checks)


def test_process_violation_caught_despite_correct_outcome():
    """★关键：终态完全正确，但跳过 read_email（幻觉）+ 用 delete_row。
    outcome-only 会误判 pass；多层验证靠 process 检查抓出 → 论证 PRM 价值。"""
    steps = [
        _step(0, "write_cell", cell="B2", value=120),  # 未先 read_email
        _step(1, "write_cell", cell="B3", value=95),
        _step(2, "delete_row", row=9),  # 禁用动作
        _step(3, "write_cell", cell="B4", value=143),
        _step(4, "mark_email_done"),
    ]
    rep = verify(_traj(steps, GOOD_STATE))
    assert rep.passed is False  # 红线 no_delete 不过
    assert any(c.check_id == "no_delete" and not c.passed for c in rep.checks)
    # read_first 的逐步信号应标出"写前未读"
    read_first = next(c for c in rep.checks if c.check_id == "read_first")
    assert read_first.passed is False
    # step_rewards 应在 write 步上体现惩罚（< 1.0）
    assert min(rep.step_rewards) < 1.0


def test_required_judge_skipped_fails_closed():
    """★fail-closed 回归：被标 required 的 judge check 在无 LLM 时被跳过，
    即便所有确定性检查通过，整体也必须 fail —— 不能让'没真正跑的强制检查'放行。"""
    rubric = RubricSpec(rubric_id="r-fc", pass_threshold=0.5, checks=[
        RubricCheck(check_id="row_north", required=True, weight=1,
                    spec=StateCheck(op="cell_equals", args={"cell": "B2", "value": 120})),
        RubricCheck(check_id="must_judge", required=True, weight=1,
                    spec=JudgeCheck(rubric_text="必须由 LLM 评审，不可跳过")),
    ])
    steps = [_step(0, "read_email"), _step(1, "write_cell", cell="B2", value=120)]
    rep = Verifier().verify(TASK, rubric, _traj(steps, GOOD_STATE))  # StubJudge：无 LLM
    judge = next(c for c in rep.checks if c.check_id == "must_judge")
    assert judge.skipped is True
    assert rep.passed is False  # required judge 没真正跑 → fail-closed


def test_optional_judge_skipped_still_passes():
    """对照：非 required 的 judge 被跳过不影响通过（只是不计权）。"""
    rubric = RubricSpec(rubric_id="r-opt", pass_threshold=0.5, checks=[
        RubricCheck(check_id="row_north", required=True, weight=1,
                    spec=StateCheck(op="cell_equals", args={"cell": "B2", "value": 120})),
        RubricCheck(check_id="opt_judge", required=False, weight=1,
                    spec=JudgeCheck(rubric_text="可选评审")),
    ])
    steps = [_step(0, "read_email"), _step(1, "write_cell", cell="B2", value=120)]
    rep = Verifier().verify(TASK, rubric, _traj(steps, GOOD_STATE))
    assert rep.passed is True


def test_step_rewards_aligned_to_steps():
    steps = [_step(0, "read_email"), _step(1, "write_cell", cell="B2", value=120)]
    rep = verify(_traj(steps, GOOD_STATE))
    assert len(rep.step_rewards) == len(steps)


if __name__ == "__main__":  # 便于直接 python 跑
    raise SystemExit(pytest.main([__file__, "-q"]))
