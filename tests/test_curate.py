"""Curate 诚实分母单测 —— 交付层最关键的不变式：

1. sft/rl/bundle 里**绝不**出现非 SIGNAL（ENV_FAULT/HARNESS_FAULT）轨迹。
2. rollout_accounting 的分母诚实自洽：基建故障被显式排除并计数，
   而合法 reward=0 负样本（COMPLETED 但低分）计入 completed，二者严格区分。
"""

from __future__ import annotations

import json
from pathlib import Path

from loom.contracts import (
    Outcome,
    RewardReport,
    RubricCheck,
    RubricSpec,
    StateCheck,
    Step,
    TaskSpec,
    Trajectory,
)
from loom.curate import curate


def _rubric_for(task: TaskSpec) -> RubricSpec:
    return RubricSpec(
        rubric_id=task.rubric_id,
        checks=[RubricCheck(check_id="c1", spec=StateCheck(op="state_equals"))],
    )


def _task(tid: str, domain: str = "spreadsheet", difficulty: str = "easy") -> TaskSpec:
    return TaskSpec(
        task_id=tid, domain=domain, difficulty=difficulty,
        instruction=f"do {tid}", env_type="mock", rubric_id=f"rub-{tid}",
    )


def _traj(tid: str, outcome: Outcome = Outcome.COMPLETED, marker: str = "") -> Trajectory:
    # marker 影响结构 hash，避免不同任务被去重误判为重复。
    return Trajectory(
        task_id=tid, policy="mock", outcome=outcome,
        steps=[Step(index=0, action={"name": "fill", "args": {"k": marker or tid}})],
        final_state={"done": True, "tag": marker or tid},
    )


def _report(tid: str, reward: float, passed: bool) -> RewardReport:
    return RewardReport(task_id=tid, trace_id=f"tr-{tid}", total_reward=reward, passed=passed)


def _records():
    # 2 个 COMPLETED 高分（应 kept）
    # 1 个 COMPLETED 低分（合法负样本：不 kept，计入 completed/legit_negatives）
    # 1 个 POLICY_ERROR（SIGNAL，但 reward 低 → 不 kept；仍是合法信号）
    # 1 个 ENV_FAULT（基建噪声 → 排除）
    # 1 个 HARNESS_FAULT（我方故障 → 排除）
    return [
        (_task("ok1"), _traj("ok1", marker="a"), _report("ok1", 0.95, True)),
        (_task("ok2"), _traj("ok2", marker="b"), _report("ok2", 0.88, True)),
        (_task("neg1"), _traj("neg1", marker="c"), _report("neg1", 0.0, False)),
        (_task("pe1"), _traj("pe1", Outcome.POLICY_ERROR, marker="d"),
         _report("pe1", 0.0, False)),
        (_task("env1"), _traj("env1", Outcome.ENV_FAULT, marker="e"),
         _report("env1", 0.0, False)),
        (_task("harn1"), _traj("harn1", Outcome.HARNESS_FAULT, marker="f"),
         _report("harn1", 0.0, False)),
    ]


def _read_task_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(json.loads(line)["task_id"])
    return ids


def test_faults_excluded_and_accounting_honest(tmp_path: Path):
    records = _records()
    manifest = curate(records, _rubric_for, out_dir=tmp_path,
                      policy_model="mock-oracle", keep_threshold=0.8)

    acc = manifest.rollout_accounting

    # —— 分母诚实 ——
    assert acc["attempted"] == 6
    assert acc["by_outcome"][Outcome.COMPLETED.value] == 3   # ok1, ok2, neg1
    assert acc["by_outcome"][Outcome.POLICY_ERROR.value] == 1
    assert acc["by_outcome"][Outcome.ENV_FAULT.value] == 1
    assert acc["by_outcome"][Outcome.HARNESS_FAULT.value] == 1
    assert acc["signal"] == 4                                 # COMPLETED + POLICY_ERROR
    assert acc["excluded_faults"] == 2                        # ENV_FAULT + HARNESS_FAULT
    assert acc["kept"] == 2                                   # ok1, ok2
    # 合法 reward=0 ≠ 基建故障：neg1 是 COMPLETED 低分 → legit_negative。
    assert acc["legit_negatives"] == 1
    assert acc["legit_negatives"] != acc["excluded_faults"]

    # —— 不变式：导出物里绝无非 SIGNAL 轨迹 ——
    sft_ids = _read_task_ids(tmp_path / "sft.jsonl")
    rl_ids = _read_task_ids(tmp_path / "rl.jsonl")
    bundle_ids = _read_task_ids(tmp_path / "bundle" / "tasks.jsonl")

    assert len(sft_ids) == acc["kept"]
    assert len(rl_ids) == acc["kept"]
    assert len(bundle_ids) == acc["kept"]

    fault_ids = {"env1", "harn1"}
    for ids in (sft_ids, rl_ids, bundle_ids):
        assert fault_ids.isdisjoint(ids), f"基建故障轨迹泄露进数据集: {set(ids) & fault_ids}"
    # 低分信号（neg1/pe1）也不该进数据集，但原因是低分，不是故障。
    assert set(sft_ids) == {"ok1", "ok2"}

    # bundle/rubrics 也只为 kept 轨迹生成。
    rubric_files = list((tmp_path / "bundle" / "rubrics").glob("*.json"))
    assert {p.stem for p in rubric_files} == {"ok1", "ok2"}

    # counts 兼容：total_in 仍 = len(records)。
    assert manifest.counts["total_in"] == 6
    assert manifest.counts["kept"] == 2


def test_accounting_merges_upstream(tmp_path: Path):
    # 调度层可把更上游的故障计数（重试耗尽隔离 / dead）merge 进诚实分母。
    records = _records()
    upstream = {"env_fault_quarantined": 3, "dead": 1}
    manifest = curate(records, _rubric_for, out_dir=tmp_path,
                      policy_model="mock-oracle", accounting=upstream)

    acc = manifest.rollout_accounting
    assert acc["env_fault_quarantined"] == 3
    assert acc["dead"] == 1
    # curate 自算的字段不被 merge 破坏。
    assert acc["signal"] == 4
    assert acc["excluded_faults"] == 2


def test_all_signal_kept_no_faults(tmp_path: Path):
    # 全是高分 COMPLETED 时：无 legit_negative、无 excluded_fault。
    records = [
        (_task("ok1"), _traj("ok1", marker="a"), _report("ok1", 0.9, True)),
        (_task("ok2"), _traj("ok2", marker="b"), _report("ok2", 0.91, True)),
    ]
    manifest = curate(records, _rubric_for, out_dir=tmp_path, policy_model="mock-oracle")
    acc = manifest.rollout_accounting
    assert acc["kept"] == 2
    assert acc["excluded_faults"] == 0
    assert acc["legit_negatives"] == 0
    assert acc["signal"] == 2
