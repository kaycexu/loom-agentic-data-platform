"""全局配置（环境变量可覆盖）。

LLM 默认走 OpenAI 兼容接口。无 API key 时，judge / LLMPolicy 自动降级到 stub，
主链路（MockPolicy + 确定性 verifier）仍然完整可跑——**不需要任何 key 即可验证全部论点**。

要开真实 LLM，三种配置方式（优先级：真实环境变量 > 项目根目录 .env > 内置默认）：
  - 最快：`export OPENAI_API_KEY=sk-...`（默认即打 api.openai.com 的 gpt-4o-mini）
  - 任意 OpenAI 兼容代理：再设 LOOM_LLM_BASE_URL / LOOM_LLM_MODEL
  - 本地持久化：把这些写进项目根目录的 .env（已 gitignore，见 .env.example）
先跑 `loom check-llm` 验证连通性，再 `loom run --policy llm`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """零依赖的极简 .env 加载器：把项目根 .env 里的 KEY=VALUE 注入环境。

    只 setdefault——**真实环境变量始终优先**，绝不覆盖。仅支持 `KEY=VALUE` 行
    （`#` 注释与空行跳过，可选前缀 `export `，去掉值两端引号）。足够本作业用，
    不引入 python-dotenv 依赖。
    """
    p = Path(path)
    if not p.is_file():
        return
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv()


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
        base_url=os.environ.get("LOOM_LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("LOOM_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("LOOM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        timeout=float(os.environ.get("LOOM_LLM_TIMEOUT", "60")),
    )


# 调度并发上限（按资源画像分级）。设计当真，演示用 asyncio 信号量实现。
DEFAULT_CONCURRENCY = {
    "light": int(os.environ.get("LOOM_CONCURRENCY_LIGHT", "128")),
    "browser_heavy": int(os.environ.get("LOOM_CONCURRENCY_BROWSER", "8")),
}
