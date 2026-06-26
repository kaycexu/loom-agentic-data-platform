"""高层编排 —— 把各模块组合成端到端命令（CLI 调用）。

数据流：Task/Rubric → Env → Rollout → Verify → (Gold/Quality) → Curate → Trace → Report。
小批生成（run/demo）走 generate_records；规模/并发（scale）走 schedule_tasks（可插拔
executor + 持久化续跑 + OTel）。
"""

from __future__ import annotations

from typing import Any, Optional

from loom.config import DEFAULT_CONCURRENCY, llm_config
from loom.contracts import RewardReport, TaskSpec, Trajectory
from loom.curate import curate
from loom.envs import make_env
from loom.obs import setup_tracing, span
from loom.quality import build_gold, evaluate_verifier, judge_variance
from loom.rollout import STRATEGIES, LLMPolicy, MockPolicy, run_rollout
from loom.schedule import schedule_tasks
from loom.tasks import canonical_tasks, generate_tasks, rubric_for
from loom.trace import RunDir, render_report
from loom.verify import Verifier, default_judge

Record = tuple[TaskSpec, Trajectory, RewardReport]
_NEG = ["missing_fill", "wrong_column", "process_violation"]


# --------------------------------------------------------------------------- #
# 轨迹生成（小批，mock-all = 每任务 4 策略）
# --------------------------------------------------------------------------- #
def _make_policy(spec: str):
    if spec == "llm":
        return LLMPolicy()
    if spec.startswith("mock:"):
        return MockPolicy(spec.split(":", 1)[1])
    if spec == "mock":
        return MockPolicy("correct")
    raise ValueError(f"未知 policy: {spec}")


def generate_records(
    tasks: list[TaskSpec], policy: str, prefer_browser: bool, verifier: Verifier
) -> tuple[list[Record], dict[str, float]]:
    if policy == "mock-all":
        specs = [(t, f"mock:{s}") for t in tasks for s in STRATEGIES]
    else:
        specs = [(t, policy) for t in tasks]

    records: list[Record] = []
    durations: dict[str, float] = {}
    with span("loom.generation", **{"loom.tasks": len(tasks), "loom.policy": policy}):
        for task, pol in specs:
            env = make_env(task.env_type, prefer_browser=prefer_browser)
            traj = run_rollout(task, env, _make_policy(pol))
            rep = verifier.verify(task, rubric_for(task), traj)
            records.append((task, traj, rep))
            durations[traj.trace_id] = traj.cost.get("latency_s", 0.0)
    return records, durations


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
def run_eval_verifier(tasks: list[TaskSpec], run: RunDir, judge_runs: int = 5,
                      otel: Optional[str] = None) -> dict[str, Any]:
    setup_tracing(otel)
    verifier = Verifier(judge=default_judge())
    gold = build_gold(tasks)
    by_id = {t.task_id: t for t in tasks}
    metrics = evaluate_verifier(verifier, gold, by_id, rubric_for)
    run.write_json("quality.json", metrics)

    jv = judge_variance(verifier, gold[0], by_id[gold[0].task_id],
                        rubric_for(by_id[gold[0].task_id]), runs=judge_runs)
    run.write_json("judge_variance.json", jv)

    records = [(by_id[g.task_id], g.trajectory, verifier.verify(by_id[g.task_id], rubric_for(by_id[g.task_id]), g.trajectory))
               for g in gold]
    run.write_records(records)
    render_report(run, title="Loom — 验证器可靠性评估",
                  subtitle="gold 集：5 任务 × (1 正确 + 3 类错误)")
    return metrics


def run_generation(tasks: list[TaskSpec], run: RunDir, policy: str = "mock-all",
                   prefer_browser: bool = False, keep_threshold: float = 0.8,
                   with_quality: bool = True, otel: Optional[str] = None) -> dict[str, Any]:
    setup_tracing(otel)
    verifier = Verifier(judge=default_judge())
    records, durations = generate_records(tasks, policy, prefer_browser, verifier)

    quality = None
    if with_quality:
        gold = build_gold(tasks)
        by_id = {t.task_id: t for t in tasks}
        quality = evaluate_verifier(verifier, gold, by_id, rubric_for)
        run.write_json("quality.json", quality)

    manifest = curate(
        records, rubric_for, out_dir=run.path("dataset"),
        policy_model=(llm_config().model if policy == "llm" else "mock-oracle"),
        keep_threshold=keep_threshold, quality_metrics=quality or {})
    run.write_records(records, durations=durations)
    render_report(run, title="Loom — agentic 数据生产 run",
                  subtitle=f"policy={policy} · tasks={len(tasks)} · 全链路")
    return {
        "rollouts": len(records),
        "passed": sum(1 for _, _, r in records if r.passed),
        "dataset_kept": manifest.counts["kept"],
        "leakage": (quality or {}).get("leakage_count"),
        "report": str(run.path("report.html")),
    }


def _scale_closures(n: int, browser_fraction: int):
    cut = int(n * browser_fraction / 100)

    def policy_for(_t: TaskSpec, i: int) -> str:
        return "mock:correct" if i % 10 < 7 else f"mock:{_NEG[i % 3]}"

    def class_for(_t: TaskSpec, i: int) -> str:
        return "browser_heavy" if i < cut else "light"

    def priority_for(t: TaskSpec, _i: int) -> int:
        return {"hard": 0, "medium": 1, "easy": 2}.get(t.difficulty, 1)

    return policy_for, class_for, priority_for


def run_scale(n: int, run: RunDir, executor: str = "async",
              caps: Optional[dict[str, int]] = None, prefer_browser: bool = False,
              browser_fraction: int = 5, store_path: Optional[str] = None,
              resume: bool = False, max_attempts: int = 2,
              otel: Optional[str] = None) -> dict[str, Any]:
    setup_tracing(otel)
    tasks = generate_tasks(n)
    policy_for, class_for, priority_for = _scale_closures(n, browser_fraction)
    res = schedule_tasks(
        tasks, policy_for=policy_for, class_for=class_for, priority_for=priority_for,
        prefer_browser=prefer_browser, executor=executor,
        caps=caps or dict(DEFAULT_CONCURRENCY), max_attempts=max_attempts,
        run_id=f"scale-{n}", store_path=store_path, resume=resume)
    run.write_json("schedule.json", res.summary)
    classes = {r.trajectory.trace_id: r.resource_class for r in res.job_results}
    durations = {r.trajectory.trace_id: r.duration_s for r in res.job_results}
    run.write_records(res.records, classes=classes, durations=durations)
    render_report(run, title="Loom — 规模 / 并发模拟",
                  subtitle=f"{n} 任务 · executor={executor} · 资源感知调度")
    return res.summary


def run_demo(run: RunDir, scale_n: int = 1000, prefer_browser: bool = False,
             executor: str = "async", otel: Optional[str] = None) -> dict[str, Any]:
    setup_tracing(otel)
    tasks = canonical_tasks()
    verifier = Verifier(judge=default_judge())

    # ① 验证器可靠性
    gold = build_gold(tasks)
    by_id = {t.task_id: t for t in tasks}
    quality = evaluate_verifier(verifier, gold, by_id, rubric_for)
    run.write_json("quality.json", quality)
    jv = judge_variance(verifier, gold[0], by_id[gold[0].task_id],
                        rubric_for(by_id[gold[0].task_id]))
    run.write_json("judge_variance.json", jv)

    # ③ 全链路生成 + 数据集（mock-all 制造对/错轨迹）
    records, durations = generate_records(tasks, "mock-all", prefer_browser, verifier)
    manifest = curate(records, rubric_for, out_dir=run.path("dataset"),
                      policy_model="mock-oracle", quality_metrics=quality)

    # ② 规模/并发（写入同一 run 的 schedule.json）
    scale_tasks = generate_tasks(scale_n)
    policy_for, class_for, priority_for = _scale_closures(scale_n, browser_fraction=5)
    sched = schedule_tasks(scale_tasks, policy_for=policy_for, class_for=class_for,
                           priority_for=priority_for, executor=executor,
                           run_id=f"demo-scale-{scale_n}")
    run.write_json("schedule.json", sched.summary)

    run.write_records(records, durations=durations)
    report = render_report(run, title="Loom — agentic 数据 + 环境生产平台 · 综合 demo",
                           subtitle="数据是产品 · 环境是验证底座 · Task+Rubric+验证是壁垒")
    return {
        "quality": {"leakage": quality["leakage_count"], "false_accept": quality["false_accept_rate"],
                    "false_reject": quality["false_reject_rate"],
                    "expected_hit": quality["expected_failed_hit_rate"]},
        "dataset_kept": manifest.counts["kept"], "dataset_in": manifest.counts["total_in"],
        "scale": {"n": scale_n, "executor": executor,
                  "throughput_per_s": sched.summary["throughput_per_s"],
                  "peak_concurrency": sched.summary["peak_concurrency"],
                  "dead_letter": sched.summary["dead_letter"]},
        "report": str(report),
    }
