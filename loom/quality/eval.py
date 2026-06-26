"""验证器可靠性评估 —— 在 gold 集上度量验证器本身。

输出（docs/design.md §6.6）：
- 混淆矩阵（accept/reject × good/bad）
- 误收率 FA（坏轨迹被判 pass）/ 误拒率 FR（好轨迹被判 fail）
- 泄露 leakage（红线坏样本被判 pass，目标 = 0）
- expected_failed 命中率（verifier 是否抓到了'预期该挂的 check'）
- judge 多次方差（仅在真实 LLM 下有意义）
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

from loom.contracts import GoldSample, RubricSpec, TaskSpec
from loom.verify import Verifier

RubricFor = Callable[[TaskSpec], RubricSpec]


def evaluate_verifier(
    verifier: Verifier,
    gold: list[GoldSample],
    tasks_by_id: dict[str, TaskSpec],
    rubric_for: RubricFor,
) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    leaks: list[str] = []
    expected_hit_num = 0
    expected_total = 0
    rows: list[dict[str, Any]] = []

    for g in gold:
        task = tasks_by_id[g.task_id]
        rep = verifier.verify(task, rubric_for(task), g.trajectory)
        predicted_good = rep.passed
        actual_good = g.should_pass

        if actual_good and predicted_good:
            tp += 1
        elif actual_good and not predicted_good:
            fn += 1
        elif (not actual_good) and predicted_good:
            fp += 1
            if g.is_redline:
                leaks.append(g.sample_id)
        else:
            tn += 1

        failed = {c.check_id for c in rep.checks if not c.passed and not c.skipped}
        hit = [c for c in g.expected_failed_checks if c in failed]
        expected_hit_num += len(hit)
        expected_total += len(g.expected_failed_checks)

        rows.append({
            "sample_id": g.sample_id,
            "negative_type": g.negative_type,
            "is_redline": g.is_redline,
            "should_pass": actual_good,
            "verifier_passed": predicted_good,
            "reward": rep.total_reward,
            "correct": actual_good == predicted_good,
            "expected_failed": g.expected_failed_checks,
            "expected_hit": hit,
            "actual_failed": sorted(failed),
        })

    n_pos = tp + fn
    n_neg = tn + fp
    n_redline = sum(1 for g in gold if g.is_redline)

    return {
        "n_samples": len(gold),
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "accuracy": round((tp + tn) / len(gold), 4) if gold else 0.0,
        "false_accept_rate": round(fp / n_neg, 4) if n_neg else 0.0,  # 误收：坏被判好
        "false_reject_rate": round(fn / n_pos, 4) if n_pos else 0.0,  # 误拒：好被判坏
        "leakage_count": len(leaks),
        "leakage_rate": round(len(leaks) / n_redline, 4) if n_redline else 0.0,
        "leakage_samples": leaks,
        "expected_failed_hit_rate": round(expected_hit_num / expected_total, 4) if expected_total else 1.0,
        "samples": rows,
    }


def judge_variance(
    verifier: Verifier,
    sample: GoldSample,
    task: TaskSpec,
    rubric: RubricSpec,
    runs: int = 5,
) -> dict[str, Any]:
    """对同一样本多次跑，统计 judge check 分数方差（评估 LLM-judge 稳定性）。"""
    scores: list[float] = []
    for _ in range(runs):
        rep = verifier.verify(task, rubric, sample.trajectory)
        for c in rep.checks:
            if c.kind == "judge" and not c.skipped:
                scores.append(c.score)
    if not scores:
        return {"runs": runs, "judge_evaluated": False, "note": "judge skipped（无 LLM）"}
    return {
        "runs": runs,
        "judge_evaluated": True,
        "scores": scores,
        "mean": round(statistics.mean(scores), 4),
        "stdev": round(statistics.pstdev(scores), 4),
    }
