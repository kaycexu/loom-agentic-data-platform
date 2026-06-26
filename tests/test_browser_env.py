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


pytestmark = pytest.mark.skipif(
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
