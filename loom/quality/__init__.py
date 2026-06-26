"""Quality / Eval 元层 —— 度量验证器本身（★面试信号最强）。"""

from loom.quality.eval import evaluate_verifier, judge_variance
from loom.quality.gold import build_gold

__all__ = ["build_gold", "evaluate_verifier", "judge_variance"]
