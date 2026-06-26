"""数据契约层 — 整条流水线的接口。

所有模块只依赖契约，不依赖彼此内部实现。
"""

from loom.contracts.models import (
    CheckResult,
    CheckSpec,
    DatasetManifest,
    GoldSample,
    JudgeCheck,
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
