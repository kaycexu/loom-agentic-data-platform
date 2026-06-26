"""Pre-action hooks —— 安全/可追溯（呼应 KOC 项目的 pre-tool-use）。

hook(action, task) -> Optional[str]：返回拦截原因则该动作被阻止（记为 error 步，不执行）。
默认数据生产链路**不启用**拦截——我们要让违规动作真实发生，从而让 Verifier 抓出来
（区别于生产部署里"阻止危险操作"的用途）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from loom.contracts import TaskSpec

PreActionHook = Callable[[dict[str, Any], TaskSpec], Optional[str]]


def block_tools(*names: str) -> PreActionHook:
    """生产部署示例：拦截高风险写工具。数据生产时一般不挂。"""
    blocked = set(names)

    def _hook(action: dict[str, Any], task: TaskSpec) -> Optional[str]:
        if (action or {}).get("name") in blocked:
            return f"高风险工具被拦截: {action.get('name')}"
        return None

    return _hook


def allowlist_tools(task: TaskSpec) -> PreActionHook:
    """只允许 task.allowed_tools 中的工具。"""
    allowed = set(task.allowed_tools)

    def _hook(action: dict[str, Any], _task: TaskSpec) -> Optional[str]:
        name = (action or {}).get("name")
        if allowed and name not in allowed:
            return f"工具不在 allowlist: {name}"
        return None

    return _hook
