"""确定性检查实现：state（作用于终态）+ process（作用于步骤）。

每个 check 返回 (aggregate CheckResult, per_step_signals)：
- aggregate 进入 RewardReport.checks，用于加权聚合与红线判定。
- per_step_signals = [(step_index, score)]，用于派生 PRM 式 step_rewards。

state 表示约定（email_to_sheet domain，见 docs/design.md §6.1）：
    {
      "sheet": {"cells": {"A2":"North","B2":120,...},
                "rows":  [{"Region":"North","Revenue":120}, ...]},
      "email": {"status": "done"|"unread"},
    }
也支持多表 {"sheets": {name: {cells, rows}}}。
"""

from __future__ import annotations

import re
from typing import Any

from loom.contracts import CheckResult, RubricCheck, Trajectory

StepSignal = tuple[int, float]  # (step_index, score)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _eq(a: Any, b: Any) -> bool:
    """带类型容忍的相等：120 == "120"，去空白比较字符串。"""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def _sheet(state: dict, name: str | None) -> dict:
    if name and isinstance(state.get("sheets"), dict):
        return state["sheets"].get(name, {}) or {}
    return state.get("sheet", {}) or {}


def _cells(state: dict, name: str | None) -> dict:
    return _sheet(state, name).get("cells", {}) or {}


def _rows(state: dict, name: str | None) -> list[dict]:
    return _sheet(state, name).get("rows", []) or []


def _resolve(state: Any, path: str) -> Any:
    cur = state
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def _is_subsequence(needle: list, haystack: list) -> bool:
    it = iter(haystack)
    return all(x in it for x in needle)


def _result(rc: RubricCheck, kind: str, score: float, passed: bool, detail: str) -> CheckResult:
    return CheckResult(
        check_id=rc.check_id,
        kind=kind,
        score=round(float(score), 4),
        passed=passed,
        weight=rc.weight,
        scope=rc.spec.scope,
        required=rc.required,
        rationale=detail,
    )


# --------------------------------------------------------------------------- #
# state checks
# --------------------------------------------------------------------------- #
def check_state(rc: RubricCheck, traj: Trajectory) -> tuple[CheckResult, list[StepSignal]]:
    st = traj.final_state
    op = rc.spec.op
    a = rc.spec.args

    if op == "cell_equals":
        got = _cells(st, a.get("sheet")).get(a["cell"])
        passed = _eq(got, a["value"])
        detail = f"{a.get('sheet','sheet')}!{a['cell']}={got!r} (期望 {a['value']!r})"
    elif op == "cell_matches":
        got = str(_cells(st, a.get("sheet")).get(a["cell"]))
        passed = bool(re.search(a["regex"], got))
        detail = f"{a['cell']}={got!r} 匹配 /{a['regex']}/ -> {passed}"
    elif op == "row_exists":
        where = a["where"]
        rows = _rows(st, a.get("sheet"))
        passed = any(all(_eq(r.get(k), v) for k, v in where.items()) for r in rows)
        detail = f"存在行 {where} -> {passed}（共 {len(rows)} 行）"
    elif op == "state_equals":
        got = _resolve(st, a["path"])
        passed = _eq(got, a["value"])
        detail = f"{a['path']}={got!r} (期望 {a['value']!r})"
    elif op == "state_contains":
        got = _resolve(st, a["path"])
        passed = (a["value"] in got) if got is not None else False
        detail = f"{a['path']} 包含 {a['value']!r} -> {passed}"
    else:  # pragma: no cover
        raise ValueError(f"未知 state op: {op}")

    return _result(rc, "state", 1.0 if passed else 0.0, passed, detail), []


# --------------------------------------------------------------------------- #
# process checks
# --------------------------------------------------------------------------- #
def check_process(rc: RubricCheck, traj: Trajectory) -> tuple[CheckResult, list[StepSignal]]:
    steps = traj.steps
    op = rc.spec.op
    a = rc.spec.args
    scope = rc.spec.scope
    names = [(s.action or {}).get("name") for s in steps]
    per_step: list[StepSignal] = []

    if op == "tool_sequence_contains":
        seq = a["sequence"]
        passed = _is_subsequence(seq, names)
        score = 1.0 if passed else 0.0
        detail = f"有序子序列 {seq} 出现于 {names} -> {passed}"

    elif op == "tool_used":
        tool = a["tool"]
        cnt = names.count(tool)
        lo, hi = a.get("min", 1), a.get("max", 10**9)
        passed = lo <= cnt <= hi
        score = 1.0 if passed else 0.0
        detail = f"{tool} 调用 {cnt} 次（区间 [{lo},{hi}]）-> {passed}"

    elif op == "forbidden_action_absent":
        bad = set(a.get("tools", []))
        if scope == "step":
            for s in steps:
                ok = (s.action or {}).get("name") not in bad
                per_step.append((s.index, 1.0 if ok else 0.0))
            passed = all(sc == 1.0 for _, sc in per_step)
        else:
            passed = not any(n in bad for n in names)
        score = (sum(sc for _, sc in per_step) / len(per_step)) if per_step else (1.0 if passed else 0.0)
        hits = [n for n in names if n in bad]
        detail = f"禁用动作 {sorted(bad)} 未出现 -> {passed}" + (f"；命中 {hits}" if hits else "")

    elif op == "tool_preceded_by":
        tool, by = a["tool"], a["by"]
        occ = [i for i, n in enumerate(names) if n == tool]
        if occ:
            for i in occ:
                ok = by in names[:i]
                per_step.append((steps[i].index, 1.0 if ok else 0.0))
            passed = all(sc == 1.0 for _, sc in per_step)
            score = sum(sc for _, sc in per_step) / len(per_step)
            detail = f"每次 {tool} 前需先 {by}；{len(occ)} 次调用，通过 {int(score*len(occ))}/{len(occ)}"
        else:
            passed, score = True, 1.0  # 未调用 tool → 空真
            detail = f"未调用 {tool}（空真通过）"

    elif op == "max_steps":
        n = a["n"]
        passed = len(steps) <= n
        score = 1.0 if passed else 0.0
        detail = f"步数 {len(steps)} <= {n} -> {passed}"

    elif op == "no_error_steps":
        if scope == "step":
            for s in steps:
                ok = not s.error
                per_step.append((s.index, 1.0 if ok else 0.0))
            passed = all(sc == 1.0 for _, sc in per_step)
            score = (sum(sc for _, sc in per_step) / len(per_step)) if per_step else 1.0
        else:
            passed = not any(s.error for s in steps)
            score = 1.0 if passed else 0.0
        errs = [s.index for s in steps if s.error]
        detail = f"无报错步 -> {passed}" + (f"；报错步 {errs}" if errs else "")

    else:  # pragma: no cover
        raise ValueError(f"未知 process op: {op}")

    return _result(rc, "process", score, passed, detail), per_step
