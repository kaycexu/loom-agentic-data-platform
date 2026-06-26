"""调度层回归 —— 并发上限、断点续跑、dead-letter、进程池、Job 可序列化、K8s manifest。"""

from __future__ import annotations

import pickle

from loom.schedule import (
    Job,
    JobResult,
    RunStore,
    execute_job,
    render_job_manifest,
    schedule_tasks,
)
from loom.tasks import canonical_tasks, generate_tasks


def _mock_correct(_t, _i):
    return "mock:correct"


def _light(_t, _i):
    return "light"


def test_async_respects_concurrency_cap():
    tasks = generate_tasks(40)
    res = schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                         caps={"light": 4}, executor="async")
    assert res.summary["total"] == 40
    assert res.summary["peak_concurrency"].get("light", 0) <= 4  # 不破上限（安全不变式）
    assert res.summary["pass_rate"] == 1.0


def test_resource_class_split():
    tasks = generate_tasks(20)
    res = schedule_tasks(tasks, policy_for=_mock_correct,
                         class_for=lambda t, i: "browser_heavy" if i < 5 else "light",
                         caps={"light": 8, "browser_heavy": 2}, executor="async")
    assert res.summary["peak_concurrency"].get("browser_heavy", 0) <= 2
    assert res.summary["by_resource_class"]["browser_heavy"] == 5


def test_store_resume_skips_completed(tmp_path):
    db = str(tmp_path / "run.db")
    tasks = generate_tasks(15)
    first = schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                           run_id="r1", store_path=db)
    assert first.summary["ran_now"] == 15
    second = schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                            run_id="r1", store_path=db, resume=True)
    assert second.summary["ran_now"] == 0
    assert second.summary["resumed_skipped"] == 15
    assert second.summary["total"] == 15  # 续跑后仍是完整 15 条


def test_dead_letter_not_silently_dropped():
    # llm policy 无 key → execute_job 抛错 → HARNESS_FAULT 重试耗尽 → dead-letter。
    # 诚实分母语义：dead 不是合法信号，结构上排除出 records，但在 fault_letter 表里可追溯。
    tasks = canonical_tasks()[:3]
    res = schedule_tasks(tasks, policy_for=lambda t, i: "llm", class_for=_light,
                         max_attempts=2, run_id="dl")
    assert res.summary["dead_letter"] == 3
    assert res.summary["passed"] == 0
    assert res.summary["completed"] == 0  # 无合法信号
    assert res.summary["pass_rate"] == 0.0  # 分母为 0 时不报错、归 0
    assert len(res.records) == 0  # dead 不进下游数据/报告（基建/我方故障结构性排除）


def test_process_executor_runs():
    tasks = generate_tasks(6)
    res = schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                         caps={"light": 2}, executor="process")
    assert res.summary["total"] == 6
    assert len(res.records) == 6
    assert res.summary["pass_rate"] == 1.0


def test_job_is_picklable():
    job = Job(run_id="x", task=canonical_tasks()[0], policy_spec="mock:correct")
    restored = pickle.loads(pickle.dumps(job))  # 进程池/Pod 需要可 pickle
    assert restored.task.task_id == job.task.task_id
    result = execute_job(restored)
    assert isinstance(result, JobResult)
    assert pickle.loads(pickle.dumps(result)).task_id == result.task_id


def test_k8s_manifest_has_resource_requests():
    job = Job(run_id="k", task=canonical_tasks()[0], resource_class="browser_heavy")
    y = render_job_manifest(job)
    assert "kind: Job" in y
    assert "2Gi" in y  # browser_heavy 的内存 request
    assert "loom/resource-class: browser_heavy" in y


# —— 诚实会计 / 成本模型 ——

def test_honest_accounting_all_signal():
    # mock:correct 全通过：每条都是合法信号，分母诚实、无故障、成本可估。
    tasks = generate_tasks(12)
    res = schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                         caps={"light": 4}, executor="async")
    s = res.summary
    assert s["attempted"] == 12
    assert s["completed"] == s["attempted"]      # 全是 SIGNAL
    assert s["signal_count"] == s["completed"]
    assert s["pass_rate"] == 1.0
    assert s["env_fault_quarantined"] == 0
    assert s["dead_letter"] == 0
    assert len(res.records) == 12                # 全部信号都进下游
    cm = s["cost_model"]
    assert cm["est_total_usd"] >= 0.0
    assert cm["est_per_1k_usd"] >= 0.0
    assert "light" in cm["by_resource_class"]


def test_honest_accounting_with_dead(tmp_path):
    # 混入 dead：llm 无 key → HARNESS_FAULT 耗尽 → status dead。
    # dead 既不进 completed/records，也不进 pass_rate 分母；fault_letter 表可追溯。
    db = str(tmp_path / "dl.db")
    tasks = canonical_tasks()[:3]
    res = schedule_tasks(tasks, policy_for=lambda t, i: "llm", class_for=_light,
                         max_attempts=2, run_id="dead", store_path=db)
    s = res.summary
    assert s["dead_letter"] == 3
    assert s["completed"] == 0                   # dead 不算 completed
    assert s["signal_count"] == 0
    assert s["pass_rate"] == 0.0                 # 全 dead：分母为 0 不报错、归 0
    assert len(res.records) == 0
    store = RunStore(db)
    try:
        assert len(store.dead_letters("dead")) == 3  # 可追溯、不静默丢弃
        assert store.quarantined("dead") == []       # 无 env 故障
    finally:
        store.close()


def test_cost_model_heterogeneous():
    # browser_heavy 与 light 混合：成本模型须同时含两类，且昂贵类单位成本主导。
    tasks = generate_tasks(20)
    res = schedule_tasks(tasks, policy_for=_mock_correct,
                         class_for=lambda t, i: "browser_heavy" if i < 6 else "light",
                         caps={"light": 8, "browser_heavy": 2}, executor="async")
    cm = res.summary["cost_model"]
    by = cm["by_resource_class"]
    assert "browser_heavy" in by and "light" in by  # 异构两类都计入
    # 昂贵类单位成本高于轻量类（成本主导项的结构性体现，与 duration 无关、确定性）。
    assert by["browser_heavy"]["unit_cost_per_min"] > by["light"]["unit_cost_per_min"]
    assert cm["est_total_usd"] >= 0.0
