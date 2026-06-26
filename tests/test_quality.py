"""Quality 元层单测 —— 验证器在 gold 集上必须 leakage=0（核心验收标准）。"""

from __future__ import annotations

from loom.quality import build_gold, evaluate_verifier
from loom.tasks import canonical_tasks, rubric_for
from loom.verify import Verifier


def test_verifier_reliability_on_gold():
    tasks = canonical_tasks()
    gold = build_gold(tasks)  # 5 任务 × 4 策略 = 20 样本
    by_id = {t.task_id: t for t in tasks}
    m = evaluate_verifier(Verifier(), gold, by_id, rubric_for)

    assert m["n_samples"] == 20
    # 红线零泄露：任何错误数据/越权轨迹都不能被判 pass
    assert m["leakage_count"] == 0
    assert m["false_accept_rate"] == 0.0
    # 正确轨迹不应被误拒
    assert m["false_reject_rate"] == 0.0
    # verifier 抓到了所有"预期该挂的 check"
    assert m["expected_failed_hit_rate"] == 1.0
    assert m["accuracy"] == 1.0
