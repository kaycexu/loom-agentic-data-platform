"""BrowserEnv smoke 测试 —— 真实 Flask + Playwright 跑通一条最小轨迹。

证明 BrowserEnv 是"重量真实"环境：起本地 Web 应用、用 headless chromium 真实操作
DOM，并且 get_state() 的形状/值与 SheetEmailEnv 逐字段一致（同一 Verifier 可直接验）。

无 playwright/flask 或无法启动 chromium 时整体 skip，保证 CI 友好。
"""

from __future__ import annotations

import pytest


def _browser_available() -> bool:
    """playwright + flask 已装，且 chromium 可启动。"""
    try:
        import flask  # noqa: F401
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(headless=True)
            b.close()
        return True
    except Exception:
        return False


# 只给「需要真实浏览器」的用例加 skip；纯逻辑用例（故障分类 helper、resource_profile）
# 必须在无 Playwright 时也跑，所以不再用模块级 pytestmark 一刀切 skip 整个文件。
requires_browser = pytest.mark.skipif(
    not _browser_available(), reason="playwright/flask 不可用或无法启动 chromium"
)

SEED = {
    "sheet": {"name": "Q2", "headers": ["Region", "Revenue"],
              "cells": {"A1": "Region", "B1": "Revenue", "A2": "North"}},
    "email": {"id": "m1", "from": "sales@corp", "subject": "Q2 revenue",
              "body": "North=120", "status": "unread"},
}


@pytest.fixture()
def env():
    from loom.envs.browser import BrowserEnv

    e = BrowserEnv()
    try:
        yield e
    finally:
        e.close()


@requires_browser
def test_reset_step_get_state(env):
    env.reset(SEED)

    # read_email：unread -> read（真实点击 #read-email）
    env.step({"name": "read_email", "args": {}})
    assert env.get_state()["email"]["status"] == "read"

    # write_cell B2=120（真实 DOM fill + change → /api/write_cell）
    env.step({"name": "write_cell", "args": {"cell": "B2", "value": 120}})

    # mark_email_done：read -> done（真实点击 #mark-done）
    env.step({"name": "mark_email_done", "args": {}})

    state = env.get_state()

    # 形状逐字段一致
    assert set(state.keys()) == {"sheet", "email"}
    assert set(state["sheet"].keys()) == {"name", "cells", "rows"}
    assert set(state["email"].keys()) == {"id", "status"}

    # 值正确（verifier 端有类型容忍：120 或 "120" 都接受）
    assert state["sheet"]["cells"]["B2"] in (120, "120")
    assert state["sheet"]["name"] == "Q2"
    assert state["email"]["id"] == "m1"
    assert state["email"]["status"] == "done"

    # rows 由 cells 的 A{n}/B{n} 派生（n>=2）
    rows = state["sheet"]["rows"]
    assert len(rows) == 1
    assert rows[0]["Region"] == "North"
    assert rows[0]["Revenue"] in (120, "120")


@requires_browser
def test_delete_row(env):
    seed = {
        "sheet": {"name": "Q2", "cells": {"A1": "Region", "B1": "Revenue",
                                          "A2": "North", "B2": 120}},
        "email": {"id": "m1", "status": "unread"},
    }
    env.reset(seed)
    assert len(env.get_state()["sheet"]["rows"]) == 1

    env.step({"name": "delete_row", "args": {"row": 2}})
    state = env.get_state()
    assert "A2" not in state["sheet"]["cells"]
    assert "B2" not in state["sheet"]["cells"]
    assert state["sheet"]["rows"] == []


def test_resource_profile():
    from loom.envs.base import BROWSER_HEAVY
    from loom.envs.browser import BrowserEnv

    assert BrowserEnv.resource_profile == BROWSER_HEAVY


# --------------------------------------------------------------------------- #
# 故障分类 helper —— 纯函数单测，不依赖 Playwright（核心不变式：基建噪声不进信号）。
# --------------------------------------------------------------------------- #
def test_classify_step_exception_tool_errors():
    """显式工具/参数校验错 → "tool"（落 obs["error"]，给策略的合法反馈）。"""
    from loom.envs.browser import _classify_step_exception

    assert _classify_step_exception(ValueError("未知工具: foo")) == "tool"
    assert _classify_step_exception(KeyError("cell")) == "tool"
    assert _classify_step_exception(TypeError("int(None)")) == "tool"


def test_classify_step_exception_faults():
    """一切非工具错（超时/HTTP/JSON 解析/任意 RuntimeError）→ "fault"（raise EnvFault）。

    这是修复点：进程是否 alive 不影响判定，未预期异常默认 fault，不被降级成 obs 反馈。"""
    import json

    from loom.envs.browser import _classify_step_exception

    assert _classify_step_exception(RuntimeError("playwright crashed")) == "fault"
    assert _classify_step_exception(TimeoutError("navigate timeout")) == "fault"
    assert _classify_step_exception(
        json.JSONDecodeError("bad json", "", 0)
    ) == "fault"
    assert _classify_step_exception(ConnectionError("flask 502")) == "fault"

    class WeirdPlaywrightError(Exception):
        pass

    assert _classify_step_exception(WeirdPlaywrightError("boom")) == "fault"


def test_classify_json_decode_error_is_fault_despite_being_valueerror():
    """JSONDecodeError 是 ValueError 子类，但解析失败属基建/数据故障 → 必须归 fault。

    若 helper 只写 isinstance(exc, _TOOL_ERRORS)，JSONDecodeError 会沿 ValueError 漏成 tool；
    helper 显式优先把它判 fault。此测试钉死该边界，防回归。"""
    import json

    from loom.envs.browser import _classify_step_exception

    assert issubclass(json.JSONDecodeError, ValueError)  # 它确实是 ValueError 子类
    assert _classify_step_exception(json.JSONDecodeError("x", "", 0)) == "fault"


# --------------------------------------------------------------------------- #
# 可选：真实浏览器在场时，验证「活着的浏览器里 _dispatch 抛非工具错 → step raise EnvFault
# → run_rollout 归 ENV_FAULT」端到端成立（基建噪声不进 reward）。
# --------------------------------------------------------------------------- #
@requires_browser
def test_live_browser_nontool_error_raises_envfault(env, monkeypatch):
    from loom.envs.base import EnvFault

    env.reset(SEED)
    assert env.is_alive()  # 进程仍健康——正是被修复的高危场景

    # 让 _dispatch 在环境健康时抛一个**非工具**异常（模拟 Playwright 超时/HTTP 失败）。
    def boom(_name, _args):
        raise RuntimeError("playwright op timed out")

    monkeypatch.setattr(env, "_dispatch", boom)

    raised = False
    try:
        env.step({"name": "read_email", "args": {}})
    except EnvFault:
        raised = True
    assert raised, "活着的浏览器里非工具异常必须 raise EnvFault，不能降级成 obs 反馈"
    assert env.is_alive()  # 环境其实没崩，但仍按 fault 处理（这正是不变式的要点）


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
