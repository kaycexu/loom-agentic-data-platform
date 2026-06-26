"""Loom CLI —— 一条命令跑全链路。

  loom demo                     一键综合 demo（验证器评估 + 全链路 + 1k 规模 → 一个看板）
  loom eval-verifier            在 gold 集上度量验证器（最强信号）
  loom run --policy mock-all    生成→验证→筛选→导出数据集 + 看板
  loom scale --n 1000           规模/并发模拟（资源感知调度）
  loom report --run out/demo    重新渲染看板
  loom materialize --n 1000     把任务+rubric 落成声明式文件
"""

from __future__ import annotations

import json

import typer
from rich import print as rprint

from loom import pipeline
from loom.tasks import load_tasks, materialize
from loom.trace import RunDir, render_report

app = typer.Typer(add_completion=False, help="Loom — agentic 数据 + 环境生产平台")


def _tasks(tasks_dir: str | None, limit: int | None):
    ts = load_tasks(tasks_dir)
    return ts[:limit] if limit else ts


@app.command()
def demo(
    out: str = typer.Option("out/demo", help="输出 run 目录"),
    scale_n: int = typer.Option(1000, help="规模模拟任务数"),
    browser: bool = typer.Option(False, "--browser", help="用真实 Playwright 浏览器环境（否则轻量降级）"),
):
    """一键综合 demo：验证器评估 + 全链路生成 + 规模模拟 → 单一看板。"""
    summary = pipeline.run_demo(RunDir(out), scale_n=scale_n, prefer_browser=browser)
    rprint(json.dumps(summary, ensure_ascii=False, indent=2))
    rprint(f"\n[bold green]看板已生成[/]: {summary['report']}")


@app.command("eval-verifier")
def eval_verifier(
    out: str = typer.Option("out/eval", help="输出 run 目录"),
    tasks_dir: str = typer.Option(None, "--tasks", help="任务目录（缺省用 canonical）"),
    judge_runs: int = typer.Option(5, help="LLM-judge 稳定性采样次数"),
):
    """在 gold 集上度量验证器本身：混淆矩阵 / 误收 / 误拒 / 泄露。"""
    metrics = pipeline.run_eval_verifier(_tasks(tasks_dir, None), RunDir(out), judge_runs=judge_runs)
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
):
    """生成→验证→筛选→导出数据集 + 看板。"""
    summary = pipeline.run_generation(
        _tasks(tasks_dir, limit), RunDir(out), policy=policy,
        prefer_browser=browser, keep_threshold=keep_threshold)
    rprint(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def scale(
    n: int = typer.Option(1000, help="任务数"),
    out: str = typer.Option("out/scale", help="输出 run 目录"),
    light: int = typer.Option(128, help="light 资源类并发上限"),
    browser_cap: int = typer.Option(8, help="browser_heavy 资源类并发上限"),
    browser_fraction: int = typer.Option(5, help="走 browser_heavy 资源类的任务占比(%)"),
):
    """规模/并发模拟：资源感知调度跑 N 个任务。"""
    summary = pipeline.run_scale(
        n, RunDir(out), concurrency={"light": light, "browser_heavy": browser_cap},
        browser_fraction=browser_fraction)
    rprint(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def report(run_dir: str = typer.Option("out/demo", "--run", help="run 目录")):
    """从已有 run 目录重新渲染看板。"""
    out = render_report(RunDir(run_dir))
    rprint(f"[bold green]看板[/]: {out}")


@app.command()
def materialize_tasks(
    out: str = typer.Option("data/tasks", "--out", help="输出目录"),
    n: int = typer.Option(None, help="生成 N 个任务（缺省=canonical 5 个）"),
):
    """把任务 + 实例化 rubric 落成声明式 JSON 文件。"""
    rprint(materialize(out, n))


if __name__ == "__main__":
    app()
