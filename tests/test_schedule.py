"""调度层回归 —— 并发上限、断点续跑、dead-letter、进程池、Job 可序列化、K8s manifest。"""

from __future__ import annotations

import pickle

from loom.schedule import (
    Job,
    JobResult,
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
    # llm policy 无 key → execute_job 抛错 → 重试耗尽 → dead-letter（仍产出占位记录）
    tasks = canonical_tasks()[:3]
    res = schedule_tasks(tasks, policy_for=lambda t, i: "llm", class_for=_light,
                         max_attempts=2, run_id="dl")
    assert res.summary["dead_letter"] == 3
    assert res.summary["passed"] == 0
    assert len(res.records) == 3  # dead 也有占位 trajectory/report，可追溯


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
