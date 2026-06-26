"""Environment 抽象 —— 验证底座的接口。

所有环境（轻量 in-memory / 重量 Playwright 浏览器 / 未来的 api/file/computer_use）
都实现同一接口，且 `get_state()` 返回同一形状的状态，供 Verifier 复用。

state 形状契约（email_to_sheet domain）：
    {
      "sheet": {"name": str,
                "cells": {"A2": "North", "B2": 120, ...},
                "rows":  [{"Region": "North", "Revenue": 120}, ...]},
      "email": {"id": str, "status": "unread"|"read"|"done"},
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

LIGHT = "light"
BROWSER_HEAVY = "browser_heavy"


@dataclass
class ToolSchema:
    name: str
    description: str
    args: dict[str, str] = field(default_factory=dict)  # 参数名 -> 说明


@runtime_checkable
class Environment(Protocol):
    resource_profile: str

    def reset(self, seed: dict[str, Any]) -> dict[str, Any]:
        """用 seed 初始化，返回初始 observation。"""
        ...

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """执行 action={'name','args'}，返回新的 observation（含 last_result）。"""
        ...

    def get_state(self) -> dict[str, Any]:
        """返回供 verifier 检查的终态快照（形状见模块 docstring）。"""
        ...

    def tools(self) -> list[ToolSchema]:
        ...

    def close(self) -> None:
        ...
