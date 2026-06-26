"""数据契约层 — 整条流水线的接口。

所有模块只依赖契约，不依赖彼此内部实现。
"""

from loom.contracts.models import (
    RETRYABLE_OUTCOMES,
    SIGNAL_OUTCOMES,
    CheckResult,
    CheckSpec,
    DatasetManifest,
    GoldSample,
    JudgeCheck,
    Outcome,
    ProcessCheck,
    RewardReport,
    RubricCheck,
    RubricSpec,
    Scope,
    StateCheck,
    Step,
    TaskSpec,
    Trajectory,
)

__all__ = [
    "CheckResult",
    "CheckSpec",
    "DatasetManifest",
    "GoldSample",
    "JudgeCheck",
    "Outcome",
    "SIGNAL_OUTCOMES",
    "RETRYABLE_OUTCOMES",
    "ProcessCheck",
    "RewardReport",
    "RubricCheck",
    "RubricSpec",
    "Scope",
    "StateCheck",
    "Step",
    "TaskSpec",
    "Trajectory",
]
