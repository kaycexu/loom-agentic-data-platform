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


# --------------------------------------------------------------------------- #
# 故障契约（fault attribution 的接口侧约定）
#
# 环境的错误分两类，必须从接口上区分开，否则基建噪声会漏进 reward 信号：
#   1) 基建/环境故障（浏览器 crash/hang、子进程退出、navigate 超时、OOM、webapp 5xx）
#      → reset()/step()/get_state() 必须 raise EnvFault。不是策略的错，应换新 env 重试。
#   2) 策略侧的工具错误（调了不存在的 tool、参数非法、越界删除被后端拒绝）
#      → 不 raise，而是作为正常 observation 返回，并在 obs["error"] 里带上原因。
#        这是给策略的合法反馈（也是合法的负信号），不应触发重试。
# --------------------------------------------------------------------------- #
class EnvFault(RuntimeError):
    """环境/基建故障 —— 语义：环境坏了，不是策略错了。

    上层（runner）捕获后把该 rollout 判为 Outcome.ENV_FAULT：换新 env 幂等重试，
    绝不计入 reward 信号；重试耗尽则进 quarantine（可追溯、不交付）。"""


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


@runtime_checkable
class EnvPool(Protocol):
    """warm 环境池 —— 重量环境（browser_heavy）的吞吐杠杆 + 崩溃驱逐点。

    起一个浏览器很贵（数百 ms–秒），所以复用底层进程；但每次 acquire 都对环境做
    全新 reset 以保证 rollout 之间状态隔离。崩溃实例必须被驱逐而非复用——这正是
    fault attribution 与资源调度的交汇点：一个 EnvFault 既要触发重试，又要把坏实例
    从池里清出去。

    acquire() -> Environment            取一个健康、已隔离的环境
    release(env, *, healthy) -> None    归还；healthy=False（曾触发 EnvFault）则销毁不复用
    close() -> None                     关闭池，回收所有底层资源
    stats: dict                         created / reused / evicted 计数（可观测）
    """

    stats: dict[str, int]

    def acquire(self) -> "Environment":
        ...

    def release(self, env: "Environment", *, healthy: bool) -> None:
        ...

    def close(self) -> None:
        ...
