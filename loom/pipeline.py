"""高层编排 —— 把各模块组合成端到端命令（CLI 调用）。

数据流：Task/Rubric → Env → Rollout → Verify → (Gold/Quality) → Curate → Trace → Report。
"""

from __future__ import annotations

from typing import Any

from loom.config import DEFAULT_CONCURRENCY, llm_config
from loom.contracts import RewardReport, TaskSpec, Trajectory
from loom.curate import curate
from loom.envs import make_env
from loom.quality import build_gold, evaluate_verifier, judge_variance
from loom.rollout import STRATEGIES, LLMPolicy, MockPolicy, run_rollout
from loom.schedule import default_classify, run_schedule
from loom.tasks import canonical_tasks, generate_tasks, load_tasks, rubric_for
from loom.trace import RunDir, render_report
from loom.verify import Verifier, default_judge

Record = tuple[TaskSpec, Trajectory, RewardReport]


# --------------------------------------------------------------------------- #
# 轨迹生成
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
    """policy ∈ {mock-all, mock, mock:<strat>, llm}。mock-all = 每任务跑 4 策略。"""
    if policy == "mock-all":
        specs = [(t, f"mock:{s}") for t in tasks for s in STRATEGIES]
    else:
        specs = [(t, policy) for t in tasks]

    records: list[Record] = []
    durations: dict[str, float] = {}
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
def run_eval_verifier(tasks: list[TaskSpec], run: RunDir, judge_runs: int = 5) -> dict[str, Any]:
    """在 gold 集上度量验证器本身（最强信号）。写 quality.json + trace + report。"""
    verifier = Verifier(judge=default_judge())
    gold = build_gold(tasks)
    by_id = {t.task_id: t for t in tasks}
    metrics = evaluate_verifier(verifier, gold, by_id, rubric_for)
    run.write_json("quality.json", metrics)

    # judge 稳定性（仅真实 LLM 有意义）
    jv = judge_variance(verifier, gold[0], by_id[gold[0].task_id], rubric_for(by_id[gold[0].task_id]), runs=judge_runs)
    run.write_json("judge_variance.json", jv)

    # 把 gold 样本的验证结果落成 trace，供看板逐条下钻
    records: list[Record] = []
    for g in gold:
        task = by_id[g.task_id]
        rep = verifier.verify(task, rubric_for(task), g.trajectory)
        records.append((task, g.trajectory, rep))
    run.write_records(records)
    render_report(run, title="Loom — 验证器可靠性评估", subtitle="gold 集：5 任务 × (1 正确 + 3 类错误)")
    return metrics


def run_generation(
    tasks: list[TaskSpec], run: RunDir, policy: str = "mock-all",
    prefer_browser: bool = False, keep_threshold: float = 0.8, with_quality: bool = True,
) -> dict[str, Any]:
    """生成→验证→(Gold/Quality)→Curate→Trace→Report。"""
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
        keep_threshold=keep_threshold, quality_metrics=quality or {},
    )
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


def run_scale(
    n: int, run: RunDir, concurrency: dict[str, int] | None = None,
    prefer_browser: bool = False, browser_fraction: int = 0,
) -> dict[str, Any]:
    """1k 规模并发模拟：MockPolicy 混合策略 + 资源感知调度。"""
    tasks = generate_tasks(n)
    concurrency = concurrency or dict(DEFAULT_CONCURRENCY)

    def strat_for(idx: int) -> str:
        return "correct" if idx % 10 < 7 else STRATEGIES[1 + (idx % 3)]

    strat_by_id = {t.task_id: strat_for(i) for i, t in enumerate(tasks)}
    # 让前 browser_fraction% 任务走 browser_heavy 资源类，演示分级限流
    heavy_cut = int(n * browser_fraction / 100)
    idx_by_id = {t.task_id: i for i, t in enumerate(tasks)}

    def rollout_fn(task: TaskSpec, attempt: int) -> Trajectory:
        env = make_env("browser", prefer_browser=prefer_browser)
        return run_rollout(task, env, MockPolicy(strat_by_id[task.task_id]), attempt=attempt)

    def classify(task: TaskSpec) -> str:
        return "browser_heavy" if idx_by_id[task.task_id] < heavy_cut else "light"

    verifier = Verifier()  # 规模演示不挂 judge（省 token）
    result = run_schedule(tasks, rollout_fn, rubric_for, verifier=verifier,
                          classify=classify, concurrency=concurrency)
    run.write_json("schedule.json", result.summary)
    classes = {s.trace_id: s.resource_class for s in result.stats}
    durations = {s.trace_id: s.duration_s for s in result.stats}
    run.write_records(result.records, classes=classes, durations=durations)
    render_report(run, title="Loom — 规模 / 并发模拟",
                  subtitle=f"{n} 任务 · 资源感知调度 · MockPolicy")
    return result.summary


def run_demo(run: RunDir, scale_n: int = 1000, prefer_browser: bool = False) -> dict[str, Any]:
    """一键综合 demo：验证器评估 + 全链路生成 + 规模模拟，合成一个看板。"""
    tasks = canonical_tasks()
    verifier = Verifier(judge=default_judge())

    # ① 验证器可靠性
    gold = build_gold(tasks)
    by_id = {t.task_id: t for t in tasks}
    quality = evaluate_verifier(verifier, gold, by_id, rubric_for)
    run.write_json("quality.json", quality)
    jv = judge_variance(verifier, gold[0], by_id[gold[0].task_id], rubric_for(by_id[gold[0].task_id]))
    run.write_json("judge_variance.json", jv)

    # ③ 全链路生成 + 数据集（mock-all 制造对/错轨迹）
    records, durations = generate_records(tasks, "mock-all", prefer_browser, verifier)
    manifest = curate(records, rubric_for, out_dir=run.path("dataset"),
                      policy_model="mock-oracle", quality_metrics=quality)

    # ② 规模/并发（写入同一 run 的 schedule.json）
    scale_tasks = generate_tasks(scale_n)

    def strat_for(idx: int) -> str:
        return "correct" if idx % 10 < 7 else STRATEGIES[1 + (idx % 3)]
    strat_by_id = {t.task_id: strat_for(i) for i, t in enumerate(scale_tasks)}
    idx_by_id = {t.task_id: i for i, t in enumerate(scale_tasks)}

    def rollout_fn(task: TaskSpec, attempt: int) -> Trajectory:
        return run_rollout(task, make_env("browser"), MockPolicy(strat_by_id[task.task_id]), attempt=attempt)

    def classify(task: TaskSpec) -> str:
        return "browser_heavy" if idx_by_id[task.task_id] < scale_n * 0.05 else "light"

    sched = run_schedule(scale_tasks, rollout_fn, rubric_for, verifier=Verifier(),
                         classify=classify)
    run.write_json("schedule.json", sched.summary)

    # 看板用全链路 records（含对/错 + check 解释）
    run.write_records(records, durations=durations)
    report = render_report(run, title="Loom — agentic 数据 + 环境生产平台 · 综合 demo",
                           subtitle="数据是产品 · 环境是验证底座 · Task+Rubric+验证是壁垒")
    return {
        "quality": {"leakage": quality["leakage_count"], "false_accept": quality["false_accept_rate"],
                    "false_reject": quality["false_reject_rate"],
                    "expected_hit": quality["expected_failed_hit_rate"]},
        "dataset_kept": manifest.counts["kept"], "dataset_in": manifest.counts["total_in"],
        "scale": {"n": scale_n, "throughput_per_s": sched.summary["throughput_per_s"],
                  "peak_concurrency": sched.summary["peak_concurrency"]},
        "report": str(report),
    }
