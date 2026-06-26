"""Scheduler —— Job 抽象 + 可插拔 Executor + 持久化/续跑 + OTel。

- jobs: 可序列化 Job / JobResult / execute_job / run_job_with_retries
- executor: AsyncExecutor（队列+优先级+分级信号量+背压）/ ProcessExecutor（真进程池）
- store: SQLite 续跑 + dead-letter
- k8s: 1 rollout = 1 Pod 的 manifest seam
- scheduler: schedule_tasks 编排
"""

from loom.schedule.executor import AsyncExecutor, Executor, ProcessExecutor, make_executor
from loom.schedule.jobs import Job, JobResult, execute_job, run_job_with_retries
from loom.schedule.k8s import render_job_manifest, render_manifests
from loom.schedule.scheduler import ScheduleResult, schedule_tasks
from loom.schedule.store import RunStore

__all__ = [
    "Job", "JobResult", "execute_job", "run_job_with_retries",
    "Executor", "AsyncExecutor", "ProcessExecutor", "make_executor",
    "RunStore", "render_job_manifest", "render_manifests",
    "ScheduleResult", "schedule_tasks",
]
