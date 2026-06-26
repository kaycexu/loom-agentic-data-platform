"""全局配置（环境变量可覆盖）。

LLM 默认走 OpenAI 兼容代理。无 API key 时，judge / LLMPolicy 自动降级到 stub，
主链路（MockPolicy + 确定性 verifier）仍然完整可跑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str | None
    timeout: float = 60.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=os.environ.get("LOOM_LLM_BASE_URL", "https://llm-proxy.tapsvc.com"),
        model=os.environ.get("LOOM_LLM_MODEL", "deepseek/deepseek-v4-flash"),
        api_key=os.environ.get("LOOM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        timeout=float(os.environ.get("LOOM_LLM_TIMEOUT", "60")),
    )


# 调度并发上限（按资源画像分级）。设计当真，演示用 asyncio 信号量实现。
DEFAULT_CONCURRENCY = {
    "light": int(os.environ.get("LOOM_CONCURRENCY_LIGHT", "128")),
    "browser_heavy": int(os.environ.get("LOOM_CONCURRENCY_BROWSER", "8")),
}
