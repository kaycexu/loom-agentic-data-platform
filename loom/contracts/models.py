"""核心数据契约（Pydantic v2）。

设计文档第 5 节的落地。关键点：
- Rubric 的 check 用 discriminated union 表达（state / process / judge），
  杜绝泛化的 `config: dict`，让"怎么验证"是声明式、可读、可复跑的。
- CheckResult 带 scope / step_index，支持 PRM 式 step-level 奖励。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

Scope = Literal["final", "step", "trajectory"]


# --------------------------------------------------------------------------- #
# Rollout 终态归因（★fault attribution 的核心枚举）
#
# 大规模跑 agentic 环境，最难的不是并发，是区分"环境坏了"和"策略错了"——
# 二者都表现为 reward=0，但含义相反。这个枚举把判别结果做成一等公民，
# 决定下游三件事：是否重试、是否计入训练信号、是否进隔离。
# --------------------------------------------------------------------------- #
class Outcome(str, Enum):
    COMPLETED = "completed"          # 跑到终态、无基建故障（含合法 reward=0）→ 合法信号
    POLICY_ERROR = "policy_error"    # 策略自身错（act 抛错 / 输出非法动作）→ 合法信号、不重试
    ENV_FAULT = "env_fault"          # 环境/基建故障（浏览器 crash/hang、子进程退出…）→ 非信号 → 重试；耗尽进 quarantine
    HARNESS_FAULT = "harness_fault"  # 我方代码/配置故障（序列化、verifier 崩、缺 key…）→ 非信号 → 重试；耗尽进 dead


#: 产生合法训练信号、不应重试的终态（reward 真实反映模型表现）
SIGNAL_OUTCOMES = frozenset({Outcome.COMPLETED, Outcome.POLICY_ERROR})
#: 基建噪声、应幂等重试的终态——绝不可漏成 reward 信号污染数据集
RETRYABLE_OUTCOMES = frozenset({Outcome.ENV_FAULT, Outcome.HARNESS_FAULT})


# --------------------------------------------------------------------------- #
# Rubric checks — 可组合的评分原子（★核心壁垒）
# --------------------------------------------------------------------------- #
class StateCheck(BaseModel):
    """作用于终态 env state 的确定性断言（高可信）。"""

    kind: Literal["state"] = "state"
    # cell_equals/cell_matches: 单元格；row_exists: 行；state_equals/contains: 任意路径
    op: Literal[
        "cell_equals",
        "cell_matches",
        "row_exists",
        "state_equals",
        "state_contains",
    ]
    args: dict[str, Any] = Field(default_factory=dict)
    scope: Scope = "final"


class ProcessCheck(BaseModel):
    """作用于轨迹步骤的检查（PRM step reward 来源）。"""

    kind: Literal["process"] = "process"
    op: Literal[
        "tool_sequence_contains",  # 有序子序列出现
        "tool_used",  # 某 tool 调用次数区间
        "forbidden_action_absent",  # 禁用动作未出现（可红线）
        "tool_preceded_by",  # 每次调用 X 前必须先有 Y（反幻觉，step 级)
        "max_steps",  # 步数预算
        "no_error_steps",  # 无报错步（step 级）
    ]
    args: dict[str, Any] = Field(default_factory=dict)
    scope: Scope = "trajectory"


class JudgeCheck(BaseModel):
    """LLM-judge，处理模糊标准，输出结构化分数 + 理由。"""

    kind: Literal["judge"] = "judge"
    rubric_text: str
    scale: int = 5
    scope: Scope = "trajectory"


CheckSpec = Annotated[
    Union[StateCheck, ProcessCheck, JudgeCheck], Field(discriminator="kind")
]


class RubricCheck(BaseModel):
    check_id: str
    weight: float = 1.0
    required: bool = False  # required 不过 → 整体强制 fail（安全红线）
    spec: CheckSpec


class RubricSpec(BaseModel):
    rubric_id: str
    version: str = "v1"
    checks: list[RubricCheck]
    aggregation: Literal["weighted_sum"] = "weighted_sum"
    pass_threshold: float = 0.8


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
class TaskSpec(BaseModel):
    task_id: str
    domain: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    instruction: str
    env_type: str = "browser"
    env_seed: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    rubric_id: str
    max_steps: int = 20
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Trajectory（commodity 产物）
# --------------------------------------------------------------------------- #
class Step(BaseModel):
    index: int
    observation: dict[str, Any] = Field(default_factory=dict)
    thought: Optional[str] = None
    action: dict[str, Any] = Field(default_factory=dict)  # {name, args}
    tool_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class Trajectory(BaseModel):
    task_id: str
    attempt: int = 0
    policy: str = "unknown"
    steps: list[Step] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    status: Literal["completed", "timeout", "error", "max_steps"] = "completed"
    # 归因结论：由 runner 在跑完后判定（环境坏了 vs 策略错了 vs 正常完成）。
    outcome: Outcome = Outcome.COMPLETED
    fault_detail: Optional[str] = None  # ENV_FAULT / POLICY_ERROR 的归因证据（如 "browser process exited"）
    cost: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""


# --------------------------------------------------------------------------- #
# Verification 结果
# --------------------------------------------------------------------------- #
class CheckResult(BaseModel):
    check_id: str
    kind: str
    score: float  # [0,1]
    passed: bool
    weight: float
    scope: Scope
    required: bool = False  # 红线 check：不过则整体 fail
    skipped: bool = False  # 无 LLM 时 judge 诚实跳过，不计入加权
    step_index: Optional[int] = None  # step 级 check 绑定到具体步
    rationale: str = ""


class RewardReport(BaseModel):
    task_id: str
    trace_id: str
    total_reward: float  # [0,1]
    passed: bool
    step_rewards: list[float] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    verifier_version: str = "v1"
    policy: str = "unknown"


# --------------------------------------------------------------------------- #
# Gold 集（度量验证器本身）
# --------------------------------------------------------------------------- #
class GoldSample(BaseModel):
    sample_id: str
    task_id: str
    trajectory: Trajectory
    should_pass: bool
    negative_type: Literal[
        "none", "missing_fill", "wrong_column", "process_violation"
    ] = "none"
    is_redline: bool = False  # 红线负样本：绝不允许被判 pass（泄露 = 0 目标）
    expected_failed_checks: list[str] = Field(default_factory=list)
    human_rationale: str = ""


# --------------------------------------------------------------------------- #
# 交付物 manifest
# --------------------------------------------------------------------------- #
class DatasetManifest(BaseModel):
    dataset_id: str
    created_at: str
    policy_model: str
    verifier_versions: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, Any] = Field(default_factory=dict)
    reward_distribution: dict[str, Any] = Field(default_factory=dict)
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    # 诚实分母：attempted / completed / policy_error / env_fault_retried / quarantined / dead，
    # 让交付数据集的"分母"可追溯——基建故障被排除而非静默漏成 reward=0 负样本。
    rollout_accounting: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
