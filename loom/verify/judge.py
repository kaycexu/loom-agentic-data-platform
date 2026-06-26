"""LLM-judge：处理模糊标准（如"是否忠实于邮件、未编造数据"）。

诚实降级：无 API key 时返回 None → 引擎把该 judge check 标为 skipped，
不计入加权（而不是假装满分），并在报告里注明。
"""

from __future__ import annotations

import json
from typing import Optional, Protocol

from loom.config import LLMConfig, llm_config
from loom.contracts import Trajectory


class JudgeClient(Protocol):
    def score(
        self, rubric_text: str, scale: int, instruction: str, traj: Trajectory
    ) -> tuple[Optional[float], str]:
        """返回 (归一化分数 [0,1] 或 None=跳过, 理由)。"""
        ...


def _summarize(traj: Trajectory, max_steps: int = 30) -> str:
    lines = [f"[最终状态] {json.dumps(traj.final_state, ensure_ascii=False)[:1500]}", "[轨迹]"]
    for s in traj.steps[:max_steps]:
        act = s.action or {}
        lines.append(f"  #{s.index} {act.get('name')}({json.dumps(act.get('args', {}), ensure_ascii=False)[:200]})")
    return "\n".join(lines)


class StubJudge:
    """无 LLM 时的占位：始终跳过，绝不伪造分数。"""

    def score(self, rubric_text, scale, instruction, traj):  # noqa: ANN001
        return None, "judge skipped（未配置 LLM_API_KEY）"


class LLMJudge:
    """OpenAI 兼容代理上的真实 judge（默认 deepseek/deepseek-v4-flash）。"""

    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI  # 延迟导入，无 llm extra 也能加载本模块

        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)

    def score(self, rubric_text, scale, instruction, traj):  # noqa: ANN001
        sys = (
            "你是严格的 agentic 轨迹评审员。只依据给定标准打分，输出 JSON："
            f'{{"score": <1..{scale} 整数>, "rationale": "<简短理由>"}}。'
        )
        user = (
            f"[任务要求]\n{instruction}\n\n[评分标准]\n{rubric_text}\n\n"
            f"{_summarize(traj)}\n\n请按标准打分。"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            raw = float(data["score"])
            norm = max(0.0, min(1.0, (raw - 1) / max(1, scale - 1)))
            return norm, f"judge={raw}/{scale}: {data.get('rationale','')}"
        except Exception as e:  # 网络/解析失败 → 跳过而非污染分数
            return None, f"judge error（跳过）: {type(e).__name__}: {e}"


def default_judge(cfg: LLMConfig | None = None) -> JudgeClient:
    cfg = cfg or llm_config()
    return LLMJudge(cfg) if cfg.enabled else StubJudge()
