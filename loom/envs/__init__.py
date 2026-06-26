"""Environment 抽象 + 实现。

- SheetEmailEnv：轻量 in-memory 参考实现（主链路 + 降级替身）。
- BrowserEnv：Playwright 驱动的最小真实 Web 应用（同接口、重量变体）。
  无 playwright/flask 时导入失败不影响主链路。
"""

from loom.envs.base import BROWSER_HEAVY, LIGHT, Environment, ToolSchema
from loom.envs.sheet_email import SheetEmailEnv

__all__ = ["Environment", "ToolSchema", "LIGHT", "BROWSER_HEAVY", "SheetEmailEnv", "make_env"]


def make_env(env_type: str, prefer_browser: bool = False) -> Environment:
    """按 env_type 造环境。browser 类型在 prefer_browser 且依赖可用时用真实浏览器，
    否则降级到 SheetEmailEnv（接口与状态形状一致，主链路不受影响）。"""
    if env_type == "browser" and prefer_browser:
        try:
            from loom.envs.browser import BrowserEnv  # 延迟导入

            return BrowserEnv()
        except Exception:
            return SheetEmailEnv()
    return SheetEmailEnv()
