"""BrowserEnv —— Playwright 驱动的最小「邮件+表格」真实 Web 应用环境。

与 SheetEmailEnv 同接口的"重量真实"变体：reset/step/get_state/tools/close
语义对齐，get_state() 返回**逐字段一致**的形状，可被同一个 Verifier 直接验证。

真实性体现：
- reset() 在子进程里起一个本地 Flask 应用（随机空闲端口），用 Playwright 启动
  headless chromium，新建**独立 browser context**（体现实例隔离），navigate 到页，
  POST /reset 注入 seed。
- step() 把 action 映射成**真实浏览器操作**：在表格 input 里 fill 单元格并触发
  change 事件、点击 #mark-done / #read-email / #delete-row-N 按钮，使状态真实改变。
- get_state() 从真实页面会话里读回（浏览器上下文内 fetch GET /state）。

健壮性：playwright / flask 未装好，或无头浏览器启动失败 → 直接抛异常，
让上层 make_env 降级到 SheetEmailEnv（不静默假装成功）。
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from loom.envs.base import BROWSER_HEAVY, ToolSchema


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BrowserEnv:
    resource_profile = BROWSER_HEAVY

    def __init__(self) -> None:
        # 依赖必须可用；缺失则抛异常（让 make_env 降级）。
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            import flask  # noqa: F401
        except Exception as e:  # pragma: no cover - 取决于环境
            raise RuntimeError(f"BrowserEnv 依赖不可用: {e}") from e

        self._proc: subprocess.Popen | None = None
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._port: int | None = None
        self._base: str | None = None

    # ----------------------------- 接口 ----------------------------- #
    def reset(self, seed: dict[str, Any]) -> dict[str, Any]:
        self._teardown()  # 幂等：重复 reset 时先清理
        self._port = _free_port()
        self._base = f"http://127.0.0.1:{self._port}"

        # 1) 起 Flask 子进程
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "loom.envs.webapp.app", "--port", str(self._port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_healthy(timeout=15.0)

        # 2) 注入 seed（真实 HTTP）
        self._http_post("/reset", seed or {})

        # 3) 起 Playwright chromium + 独立 context
        try:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
            self._context = self._browser.new_context()  # 独立隔离上下文
            self._page = self._context.new_page()
            self._page.goto(self._base, wait_until="networkidle")
        except Exception as e:
            self._teardown()
            raise RuntimeError(f"Playwright 启动失败: {e}") from e

        return self._observe("已重置浏览器环境")

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        name = (action or {}).get("name")
        args = (action or {}).get("args", {}) or {}
        try:
            result = self._dispatch(name, args)
            return self._observe(result)
        except Exception as e:
            obs = self._observe(f"error: {e}")
            obs["error"] = str(e)
            return obs

    def get_state(self) -> dict[str, Any]:
        """从真实页面会话里读回状态（浏览器上下文内 fetch /state）。"""
        state = self._page_fetch_state()
        # 形状与 SheetEmailEnv.get_state() 逐字段一致。
        return {
            "sheet": {
                "name": state["sheet"]["name"],
                "cells": dict(state["sheet"]["cells"]),
                "rows": list(state["sheet"]["rows"]),
            },
            "email": {"id": state["email"]["id"], "status": state["email"]["status"]},
        }

    def tools(self) -> list[ToolSchema]:
        # 与 SheetEmailEnv 完全一致的工具集。
        return [
            ToolSchema("read_email", "读取收件箱里的邮件正文", {"email_id": "可选，邮件 id"}),
            ToolSchema("read_sheet", "读取表格当前内容", {}),
            ToolSchema("write_cell", "写入单元格", {"cell": "如 B2", "value": "值"}),
            ToolSchema("mark_email_done", "把邮件标记为已处理", {"email_id": "可选"}),
            ToolSchema("delete_row", "删除某一行（高风险）", {"row": "行号"}),
        ]

    def close(self) -> None:
        self._teardown()

    # ----------------------------- 动作分发（真实浏览器操作） ----------------------------- #
    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        page = self._page
        if page is None:
            raise RuntimeError("环境未 reset")

        if name == "read_email":
            page.click("#read-email")
            self._settle()
            return "已阅读邮件"

        if name == "read_sheet":
            # 真实读取：从页面读回单元格快照。
            state = self._page_fetch_state()
            return f"表格: {state['sheet']['cells']}"

        if name == "write_cell":
            cell, value = args["cell"], args.get("value")
            self._write_cell(cell, value)
            return f"已写入 {cell}={value}"

        if name == "mark_email_done":
            page.click("#mark-done")
            self._settle()
            return "邮件已标记为 done"

        if name == "delete_row":
            row = int(args["row"])
            sel = f"#delete-row-{row}"
            if page.query_selector(sel) is not None:
                page.click(sel)
                self._settle()
            else:
                # 行不在 DOM（如越界）→ 仍走真实后端 HTTP，使语义与 SheetEmailEnv 对齐。
                self._page_fetch("/api/delete_row", {"row": row})
            return f"已删除第 {row} 行"

        raise ValueError(f"未知工具: {name}")

    def _write_cell(self, cell: str, value: Any) -> None:
        """优先真实 DOM 操作：fill 对应 input 并触发 change；input 不存在则走真实后端。"""
        page = self._page
        sel = f"#cell-{cell}"
        el = page.query_selector(sel)
        if el is not None and (el.evaluate("e => e.tagName") == "INPUT"):
            page.fill(sel, "" if value is None else str(value))
            # 触发 change → 前端 fetch /api/write_cell（真实浏览器交互链路）。
            page.dispatch_event(sel, "change")
            self._settle()
        else:
            # 目标行尚未渲染（如首次写更靠后的行）→ 经由浏览器上下文发真实 HTTP。
            self._page_fetch("/api/write_cell", {"cell": cell, "value": value})

    # ----------------------------- 状态读取 ----------------------------- #
    def _page_fetch_state(self) -> dict[str, Any]:
        return self._page_fetch_get("/state")

    def _settle(self) -> None:
        # 等前端 fetch 回填完成。
        try:
            self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

    def _page_fetch(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """在浏览器上下文里发 POST（真实页面会话），返回解析后的 JSON。"""
        result = self._page.evaluate(
            """async ({path, body}) => {
                const res = await fetch(path, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body || {}),
                });
                return await res.json();
            }""",
            {"path": path, "body": body},
        )
        return result

    def _page_fetch_get(self, path: str) -> dict[str, Any]:
        return self._page.evaluate(
            """async (path) => {
                const res = await fetch(path);
                return await res.json();
            }""",
            path,
        )

    def _observe(self, message: str) -> dict[str, Any]:
        """observation 形状对齐 SheetEmailEnv._observe（inbox / sheet / last_result）。"""
        state = self._page_fetch_state()
        cells = state["sheet"]["cells"]
        return {
            "inbox": [{"id": state["email"]["id"], "status": state["email"]["status"]}],
            "sheet": {"name": state["sheet"]["name"], "headers": ["Region", "Revenue"], "cells": dict(cells)},
            "last_result": message,
        }

    # ----------------------------- HTTP / 进程管理 ----------------------------- #
    def _wait_healthy(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("Flask 子进程提前退出")
            try:
                self._http_get("/healthz")
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("Flask 应用启动超时")

    def _http_get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self._base + path, timeout=3) as r:
            return json.loads(r.read().decode())

    def _http_post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self._base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())

    def _teardown(self) -> None:
        for closer in (
            lambda: self._page and self._page.close(),
            lambda: self._context and self._context.close(),
            lambda: self._browser and self._browser.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self._page = self._context = self._browser = self._pw = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
