"""Loom CLI —— 一条命令跑全链路。

  loom demo                         一键综合 demo（验证器评估 + 全链路 + 1k 规模 → 一个看板）
  loom eval-verifier                在 gold 集上度量验证器（最强信号）
  loom run --policy mock-all        生成→验证→筛选→导出数据集 + 看板
  loom scale --n 1000 --executor async|process   规模/并发（资源感知 + 续跑 + dead-letter）
  loom report --run out/demo        重新渲染看板
  loom materialize-tasks --n 1000   把任务+rubric 落成声明式文件
  loom k8s-manifest --n 3           生成 "1 rollout=1 Pod" 的样例 manifest
  loom run-job --task-id <id>       单个 rollout（K8s Pod 容器入口）

可观测：所有命令支持 --otel off|console|otlp（otlp 经 OTEL_EXPORTER_OTLP_ENDPOINT → Jaeger）。
"""

from __future__ import annotations

import json

import typer
from rich import print as rprint

from loom import pipeline
from loom.tasks import canonical_tasks, generate_tasks, load_tasks, materialize
from loom.trace import RunDir, render_report

app = typer.Typer(add_completion=False, help="Loom — agentic 数据 + 环境生产平台")


def _tasks(tasks_dir, limit):
    ts = load_tasks(tasks_dir)
    return ts[:limit] if limit else ts


@app.command()
def demo(
    out: str = typer.Option("out/demo", help="输出 run 目录"),
    scale_n: int = typer.Option(1000, help="规模模拟任务数"),
    browser: bool = typer.Option(False, "--browser", help="用真实 Playwright 浏览器环境"),
    executor: str = typer.Option("async", help="规模阶段 executor：async | process"),
    otel: str = typer.Option("off", help="链路追踪：off | console | otlp"),
):
    """一键综合 demo：验证器评估 + 全链路生成 + 规模模拟 → 单一看板。"""
    summary = pipeline.run_demo(RunDir(out), scale_n=scale_n, prefer_browser=browser,
                                executor=executor, otel=otel)
    rprint(json.dumps(summary, ensure_ascii=False, indent=2))
    rprint(f"\n[bold green]看板已生成[/]: {summary['report']}")


@app.command("eval-verifier")
def eval_verifier(
    out: str = typer.Option("out/eval", help="输出 run 目录"),
    tasks_dir: str = typer.Option(None, "--tasks", help="任务目录（缺省用 canonical）"),
    judge_runs: int = typer.Option(5, help="LLM-judge 稳定性采样次数"),
    otel: str = typer.Option("off", help="off | console | otlp"),
):
    """在 gold 集上度量验证器本身：混淆矩阵 / 误收 / 误拒 / 泄露。"""
    metrics = pipeline.run_eval_verifier(_tasks(tasks_dir, None), RunDir(out),
                                         judge_runs=judge_runs, otel=otel)
    rprint(json.dumps({k: v for k, v in metrics.items() if k != "samples"}, ensure_ascii=False, indent=2))
    color = "green" if metrics["leakage_count"] == 0 else "red"
    rprint(f"\n[bold {color}]红线泄露 = {metrics['leakage_count']}[/]（误收率 {metrics['false_accept_rate']}）")


@app.command()
def run(
    out: str = typer.Option("out/run", help="输出 run 目录"),
    policy: str = typer.Option("mock-all", help="mock-all | mock | mock:<strat> | llm"),
    tasks_dir: str = typer.Option(None, "--tasks", help="任务目录（缺省用 canonical）"),
    limit: int = typer.Option(None, help="只跑前 N 个任务"),
    browser: bool = typer.Option(False, "--browser", help="用真实浏览器环境"),
    keep_threshold: float = typer.Option(0.8, help="数据集保留的 reward 阈值"),
    otel: str = typer.Option("off", help="off | console | otlp"),
):
    """生成→验证→筛选→导出数据集 + 看板。"""
    summary = pipeline.run_generation(_tasks(tasks_dir, limit), RunDir(out), policy=policy,
                                      prefer_browser=browser, keep_threshold=keep_threshold, otel=otel)
    rprint(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def scale(
    n: int = typer.Option(1000, help="任务数"),
    out: str = typer.Option("out/scale", help="输出 run 目录"),
    executor: str = typer.Option("async", help="async（队列+信号量）| process（真进程池隔离）"),
    light: int = typer.Option(128, help="light 资源类并发上限"),
    browser_cap: int = typer.Option(8, help="browser_heavy 资源类并发上限"),
    browser_fraction: int = typer.Option(5, help="走 browser_heavy 资源类的任务占比(%)"),
    store: str = typer.Option(None, "--store", help="SQLite run store 路径（启用持久化/续跑）"),
    resume: bool = typer.Option(False, "--resume", help="按 store 跳过已完成任务（断点续跑）"),
    max_attempts: int = typer.Option(2, help="单任务最大尝试次数（含重试）"),
    otel: str = typer.Option("off", help="off | console | otlp"),
):
    """规模/并发模拟：资源感知调度 + 可插拔 executor + 持久化续跑 + dead-letter。"""
    summary = pipeline.run_scale(
        n, RunDir(out), executor=executor,
        caps={"light": light, "browser_heavy": browser_cap},
        browser_fraction=browser_fraction, store_path=store, resume=resume,
        max_attempts=max_attempts, otel=otel)
    rprint(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def report(run_dir: str = typer.Option("out/demo", "--run", help="run 目录")):
    """从已有 run 目录重新渲染看板。"""
    rprint(f"[bold green]看板[/]: {render_report(RunDir(run_dir))}")


@app.command("materialize-tasks")
def materialize_tasks(
    out: str = typer.Option("data/tasks", "--out", help="输出目录"),
    n: int = typer.Option(None, help="生成 N 个任务（缺省=canonical 5 个）"),
):
    """把任务 + 实例化 rubric 落成声明式 JSON 文件。"""
    rprint(materialize(out, n))


@app.command("k8s-manifest")
def k8s_manifest(
    n: int = typer.Option(3, help="生成前 N 个 rollout 的 Job manifest 样例"),
    out: str = typer.Option("deploy/k8s", "--out", help="manifest 输出目录"),
    image: str = typer.Option("loom:latest", help="容器镜像"),
):
    """渲染 '1 rollout = 1 K8s Job/Pod' 的样例 manifest（横向扩展 seam）。"""
    from loom.schedule import Job, render_manifests

    tasks = generate_tasks(max(n, 5))
    jobs = [Job(run_id="k8s", task=t, resource_class=("browser_heavy" if i == 0 else "light"))
            for i, t in enumerate(tasks)]
    rprint(render_manifests(jobs, out, limit=n, image=image))


@app.command("run-job")
def run_job(
    task_id: str = typer.Option(..., "--task-id", help="任务 id（确定性重建）"),
    policy: str = typer.Option("mock:correct", help="policy spec"),
    browser: bool = typer.Option(False, "--browser"),
    pool: int = typer.Option(1000, help="重建任务时的生成池大小"),
):
    """执行单个 rollout —— K8s Pod 容器入口（OTel 经 LOOM_OTEL/OTLP 配置）。"""
    from loom.obs import setup_tracing
    from loom.schedule import Job, execute_job

    setup_tracing()  # 读 LOOM_OTEL 环境变量
    pool_tasks = {t.task_id: t for t in generate_tasks(pool)}
    pool_tasks.update({t.task_id: t for t in canonical_tasks()})
    task = pool_tasks.get(task_id)
    if task is None:
        raise typer.BadParameter(f"未找到任务 {task_id}")
    res = execute_job(Job(run_id="single", task=task, policy_spec=policy, prefer_browser=browser))
    rprint(json.dumps({"task_id": res.task_id, "status": res.status, "passed": res.report.passed,
                       "reward": res.report.total_reward, "steps": len(res.trajectory.steps)},
                      ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
