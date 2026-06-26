"""Curator —— 验证过的 rollout → 可交付数据集（SFT / RL / Task+Rubric bundle）。"""

from loom.curate.curator import Record, curate

__all__ = ["Record", "curate"]
