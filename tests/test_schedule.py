"""调度层回归 —— 并发上限、断点续跑、dead-letter、进程池、Job 可序列化、K8s manifest。"""

from __future__ import annotations

import pickle
import time

import pytest

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


# —— 重试会计不被折叠（medium-1）——

def test_retry_attempts_not_collapsed_in_accounting():
    # HARNESS 耗尽（llm 无 key）会重试到 max_attempts；rollout_attempts 须计入每次 attempt，
    # 不被折叠成 task 数。attempts_detail/total_duration_s 也逐次累计。
    tasks = canonical_tasks()[:3]
    res = schedule_tasks(tasks, policy_for=lambda t, i: "llm", class_for=_light,
                         max_attempts=2, run_id="retry")
    s = res.summary
    assert s["attempted"] == 3                       # task 数（折叠视图）
    assert s["rollout_attempts"] == 6                # 3 task × 2 attempt（真实执行次数）
    assert s["rollout_attempts"] > s["attempted"]    # 重试没被会计折叠隐藏
    for r in res.job_results:
        assert r.attempts == 2
        assert len(r.attempts_detail) == 2           # 每次 attempt 一条明细
        assert all(d["outcome"] == "harness_fault" for d in r.attempts_detail)
        # total 是全部 attempt 之和，≥ 末次 duration（折叠口径会低报）
        assert r.total_duration_s >= r.duration_s


def test_cost_model_counts_all_attempts(monkeypatch):
    # 用可控 sleep 的 fake execute_job 走 HARNESS 重试路径：成本须按 total_duration_s
    # （全部 attempt 之和）计，严格 > 仅算单次 duration_s 的折叠口径。
    import loom.schedule.jobs as jobs_mod

    sleep_s = 0.01

    def _slow_harness(job):
        time.sleep(sleep_s)
        raise RuntimeError("injected harness fault for cost accounting")

    monkeypatch.setattr(jobs_mod, "execute_job", _slow_harness)
    tasks = canonical_tasks()[:2]
    res = schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                         max_attempts=3, run_id="cost", executor="async")
    s = res.summary
    assert s["rollout_attempts"] == 6                # 2 task × 3 attempt
    # 成本口径（total_duration_s 之和）≈ 6 × sleep；远大于只算末次（2 × sleep）的折叠口径。
    total_minutes = sum(r.total_duration_s for r in res.job_results) / 60.0
    last_only_minutes = sum(r.duration_s for r in res.job_results) / 60.0
    cm = s["cost_model"]
    light_minutes = cm["by_resource_class"]["light"]["minutes"]
    assert light_minutes > last_only_minutes  # 计入了多次 attempt（非折叠口径）
    # cost_model 的分钟数 == 全部 attempt 的 total_duration_s 之和（容忍 round(,4) 舍入）。
    assert light_minutes == pytest.approx(total_minutes, abs=1e-4)
    assert cm["est_total_usd"] > 0.0


# —— fault_letter 生命周期：resume 成功后清旧行（medium-2）——

def test_resume_success_clears_stale_fault_letter(tmp_path):
    db = str(tmp_path / "lifecycle.db")
    tasks = canonical_tasks()[:3]
    target = tasks[0].task_id

    # run1：全部用 llm（无 key）→ HARNESS dead，task[0] 进 fault_letter。
    schedule_tasks(tasks, policy_for=lambda t, i: "llm", class_for=_light,
                   max_attempts=2, run_id="lc", store_path=db)
    store = RunStore(db)
    try:
        assert any(d["task_id"] == target for d in store.dead_letters("lc"))  # run1 后确在 dead 表
        assert target not in store.completed_task_ids("lc")
    finally:
        store.close()

    # run2：同 run_id resume，改 mock:correct → 全部成功。dead 不是 completed，故会被重跑。
    schedule_tasks(tasks, policy_for=_mock_correct, class_for=_light,
                   run_id="lc", store_path=db, resume=True)
    store = RunStore(db)
    try:
        # 旧 fault_letter 行已清：不再把已完成 task 报告为 dead（审计自洽）。
        assert all(d["task_id"] != target for d in store.dead_letters("lc"))
        assert store.dead_letters("lc") == []
        assert target in store.completed_task_ids("lc")
    finally:
        store.close()
