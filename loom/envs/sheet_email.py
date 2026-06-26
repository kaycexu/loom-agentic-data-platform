"""SheetEmailEnv —— 轻量 in-memory 环境（参考实现 + mock 降级）。

模拟一个"邮件收件箱 + 表格"应用：agent 读邮件、把数据写进表格、标记邮件已处理。
这是确定性、零依赖的环境，用于：
1. MockPolicy 的主链路（造对/错轨迹喂 verifier）；
2. 无 Playwright/无头环境时 BrowserEnv 的降级替身（get_state 形状完全一致）。

BrowserEnv（loom/envs/browser.py）是同接口的"重量真实"变体，行为与本类对齐。
"""

from __future__ import annotations

import copy
from typing import Any

from loom.envs.base import LIGHT, ToolSchema


def _rows_from_cells(cells: dict[str, Any]) -> list[dict[str, Any]]:
    """A{n}->Region, B{n}->Revenue（n>=2，1 为表头），派生结构化行。"""
    rows = []
    n = 2
    while f"A{n}" in cells or f"B{n}" in cells:
        region, revenue = cells.get(f"A{n}"), cells.get(f"B{n}")
        if region is not None or revenue is not None:
            rows.append({"Region": region, "Revenue": revenue})
        n += 1
    return rows


class SheetEmailEnv:
    resource_profile = LIGHT

    def __init__(self) -> None:
        self._sheet_name = "Q2"
        self._headers = ["Region", "Revenue"]
        self._cells: dict[str, Any] = {}
        self._email: dict[str, Any] = {}

    # ----------------------------- 接口 ----------------------------- #
    def reset(self, seed: dict[str, Any]) -> dict[str, Any]:
        seed = copy.deepcopy(seed or {})
        sheet = seed.get("sheet", {})
        self._sheet_name = sheet.get("name", "Q2")
        self._headers = sheet.get("headers", ["Region", "Revenue"])
        self._cells = dict(sheet.get("cells", {"A1": "Region", "B1": "Revenue"}))
        email = seed.get("email", {})
        self._email = {"id": email.get("id", "m1"),
                       "from": email.get("from", "sales@corp"),
                       "subject": email.get("subject", ""),
                       "body": email.get("body", ""),
                       "status": email.get("status", "unread")}
        return self._observe("已重置环境")

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        name = (action or {}).get("name")
        args = (action or {}).get("args", {}) or {}
        try:
            result = self._dispatch(name, args)
            return self._observe(result)
        except Exception as e:  # 工具内部错误 → 观测里带 error
            obs = self._observe(f"error: {e}")
            obs["error"] = str(e)
            return obs

    def get_state(self) -> dict[str, Any]:
        return {
            "sheet": {"name": self._sheet_name, "cells": dict(self._cells),
                      "rows": _rows_from_cells(self._cells)},
            "email": {"id": self._email.get("id"), "status": self._email.get("status")},
        }

    def tools(self) -> list[ToolSchema]:
        return [
            ToolSchema("read_email", "读取收件箱里的邮件正文", {"email_id": "可选，邮件 id"}),
            ToolSchema("read_sheet", "读取表格当前内容", {}),
            ToolSchema("write_cell", "写入单元格", {"cell": "如 B2", "value": "值"}),
            ToolSchema("mark_email_done", "把邮件标记为已处理", {"email_id": "可选"}),
            ToolSchema("delete_row", "删除某一行（高风险）", {"row": "行号"}),
        ]

    def close(self) -> None:  # in-memory 无需清理
        pass

    # ----------------------------- 内部 ----------------------------- #
    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "read_email":
            self._email["status"] = "read" if self._email["status"] == "unread" else self._email["status"]
            return f"邮件正文: {self._email.get('body','')}"
        if name == "read_sheet":
            return f"表格: {self._cells}"
        if name == "write_cell":
            cell, value = args["cell"], args.get("value")
            self._cells[cell] = value
            return f"已写入 {cell}={value}"
        if name == "mark_email_done":
            self._email["status"] = "done"
            return "邮件已标记为 done"
        if name == "delete_row":
            row = int(args["row"])
            for col in ("A", "B"):
                self._cells.pop(f"{col}{row}", None)
            return f"已删除第 {row} 行"
        raise ValueError(f"未知工具: {name}")

    def _observe(self, message: str) -> dict[str, Any]:
        return {
            "inbox": [{"id": self._email.get("id"), "from": self._email.get("from"),
                       "subject": self._email.get("subject"), "status": self._email.get("status"),
                       "body": self._email.get("body")}],
            "sheet": {"name": self._sheet_name, "headers": self._headers, "cells": dict(self._cells)},
            "last_result": message,
        }
