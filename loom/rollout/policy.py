"""Policy —— 产出轨迹的策略。

- MockPolicy（主链路）：脚本化 oracle，按策略制造对/错轨迹，专门喂给 Verifier 区分。
  四种策略对应 gold 集的负样本 taxonomy：
    correct / missing_fill / wrong_column / process_violation
- LLMPolicy（optional 佐证）：真模型，结构化 JSON 动作协议（跨 provider 稳健），
  维持 scratchpad 做长程记忆。无 LLM key 时不可用（CLI 层降级到 mock）。
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from loom.config import LLMConfig, llm_config
from loom.contracts import TaskSpec
from loom.envs import EnvFault, ToolSchema

STRATEGIES = ["correct", "missing_fill", "wrong_column", "process_violation"]


class Policy(Protocol):
    name: str

    def reset(self, task: TaskSpec, tools: list[ToolSchema]) -> None: ...

    def act(self, observation: dict[str, Any]) -> Optional[dict[str, Any]]:
        """返回 {'name','args','thought'?}；None 表示结束。"""
        ...


# --------------------------------------------------------------------------- #
# MockPolicy —— 主链路
# --------------------------------------------------------------------------- #
class MockPolicy:
    def __init__(self, strategy: str = "correct"):
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略 {strategy}，可选 {STRATEGIES}")
        self.strategy = strategy
        self.name = f"mock:{strategy}"
        self._queue: list[dict[str, Any]] = []
        self.usage: dict[str, Any] = {}

    def reset(self, task: TaskSpec, tools: list[ToolSchema]) -> None:
        truth: dict[str, Any] = task.metadata["truth"]
        cells = sorted(truth.items())  # [(B2,120),(B3,95),...]
        self._queue = self._plan(cells)

    def act(self, observation: dict[str, Any]) -> Optional[dict[str, Any]]:
        return self._queue.pop(0) if self._queue else None

    def _plan(self, cells: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        read = {"name": "read_email", "args": {}}
        done = {"name": "mark_email_done", "args": {}}
        writes = [{"name": "write_cell", "args": {"cell": c, "value": v}} for c, v in cells]

        if self.strategy == "correct":
            return [read, *writes, done]

        if self.strategy == "missing_fill":  # 漏填最后一格
            return [read, *writes[:-1], done]

        if self.strategy == "wrong_column":  # 错位/填错值（保证确实错）
            vals = [w["args"]["value"] for w in writes]
            bad_vals = list(vals)
            if len(set(vals)) >= 2:
                bad_vals[0], bad_vals[1] = bad_vals[1], bad_vals[0]  # 交换前两格
            else:
                try:
                    bad_vals[-1] = bad_vals[-1] + 1
                except TypeError:
                    bad_vals[-1] = f"{bad_vals[-1]}_x"
            bad = [{"name": "write_cell", "args": {"cell": w["args"]["cell"], "value": bad_vals[i]}}
                   for i, w in enumerate(writes)]
            return [read, *bad, done]

        # process_violation：终态正确，但未先读邮件(反幻觉违规) + 调用禁用动作 delete_row
        return [*writes, {"name": "delete_row", "args": {"row": 99}}, done]


# --------------------------------------------------------------------------- #
# LLMPolicy —— optional 真模型佐证
# --------------------------------------------------------------------------- #
class LLMPolicy:
    def __init__(self, cfg: LLMConfig | None = None, max_history: int = 12):
        self.cfg = cfg or llm_config()
        if not self.cfg.enabled:
            raise RuntimeError("LLMPolicy 需要 LLM_API_KEY；无 key 请用 MockPolicy")
        from openai import OpenAI

        self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key, timeout=self.cfg.timeout)
        self.name = f"llm:{self.cfg.model}"
        self.max_history = max_history
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def reset(self, task: TaskSpec, tools: list[ToolSchema]) -> None:
        self.task = task
        self.tools = tools
        self.scratch: list[str] = []  # 长程记忆：已做过的动作摘要

    def act(self, observation: dict[str, Any]) -> Optional[dict[str, Any]]:
        sys = self._system_prompt()
        user = self._user_prompt(observation)
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model, temperature=0.0,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            if resp.usage:
                self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            # LLM 代理故障（超时/限流/鉴权/网络/响应非 JSON）= 上游基建噪声，不是模型的合法决策。
            # 绝不能静默返回 "submit"（=_DONE_NAMES）把它伪装成"模型跑完了"的 COMPLETED 负样本——
            # 那会让一次 API 抖动漏成训练信号。抛 EnvFault → runner 归 ENV_FAULT（重试；耗尽 quarantine），
            # 结构上排除出数据集。只有模型主动 {"done":true} 才算合法结束。
            raise EnvFault(f"LLM proxy fault: {type(e).__name__}: {e}") from e

        if data.get("done"):
            return None
        action = {"name": data.get("tool"), "args": data.get("args", {}) or {}, "thought": data.get("thought", "")}
        self.scratch.append(f"{action['name']}({action['args']})")
        self.scratch = self.scratch[-self.max_history:]
        return action

    def _system_prompt(self) -> str:
        tool_lines = "\n".join(f"  - {t.name}: {t.description} args={t.args}" for t in self.tools)
        return (
            "你是一个在'邮件+表格'应用里操作的 agent。每步只输出一个 JSON 动作：\n"
            '  {"thought":"...","tool":"<工具名>","args":{...}}  完成时输出 {"done":true}\n'
            f"可用工具：\n{tool_lines}\n"
            "规则：先 read_email 理解要求，再把每个地区的营收 write_cell 到对应行的 B 列，"
            "最后 mark_email_done。不要调用 delete_row。不要编造数据。"
        )

    def _user_prompt(self, obs: dict[str, Any]) -> str:
        return (
            f"[任务]\n{self.task.instruction}\n\n"
            f"[当前观测]\n{json.dumps(obs, ensure_ascii=False)[:2000]}\n\n"
            f"[已做动作]\n{self.scratch}\n\n请输出下一个 JSON 动作。"
        )
