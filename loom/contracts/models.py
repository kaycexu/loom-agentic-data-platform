"""核心数据契约（Pydantic v2）。

设计文档第 5 节的落地。关键点：
- Rubric 的 check 用 discriminated union 表达（state / process / judge），
  杜绝泛化的 `config: dict`，让"怎么验证"是声明式、可读、可复跑的。
- CheckResult 带 scope / step_index，支持 PRM 式 step-level 奖励。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

Scope = Literal["final", "step", "trajectory"]


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
    provenance: dict[str, Any] = Field(default_factory=dict)
