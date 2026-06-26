"""可插拔 Executor —— 同一组 Job 可在不同后端上跑，体现 rollout 隔离的层级。

- AsyncExecutor：单进程 asyncio，优先级队列 + N worker + 分级信号量 + 背压；
  in-process 线程隔离，OTel 全 span 树。默认。
- ProcessExecutor：每个资源类一个进程池（max_workers=该类并发上限），真 OS 进程隔离 + 多核；
  子进程经 initializer 配 OTel，trace 经 carrier 跨进程链接。
- K8s 映射：见 loom/schedule/k8s.py（把 Job 渲染成 1 Pod/rollout 的 manifest）。

两者签名一致：run(jobs, caps, max_attempts, store, on_result) -> (results, peak_concurrency)。
重试/退避/dead-letter 在 jobs.run_job_with_retries 内，两后端共用。
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Protocol

from loom.schedule.jobs import Job, JobResult, run_job_with_retries
from loom.schedule.store import RunStore

OnResult = Callable[[JobResult], None]


class Executor(Protocol):
    name: str

    def run(
        self, jobs: list[Job], caps: dict[str, int], max_attempts: int,
        store: Optional[RunStore] = None, on_result: Optional[OnResult] = None,
    ) -> tuple[list[JobResult], dict[str, int]]:
        ...


class AsyncExecutor:
    name = "async"

    def run(self, jobs, caps, max_attempts, store=None, on_result=None):
        return asyncio.run(self._arun(jobs, caps, max_attempts, store, on_result))

    async def _arun(self, jobs, caps, max_attempts, store, on_result):
        sems = {cls: asyncio.Semaphore(c) for cls, c in caps.items()}
        active: Counter = Counter()
        peak: Counter = Counter()
        results: list[JobResult] = []
        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        # 专用线程池，容量 = 各资源类上限之和。否则 asyncio.to_thread 落到默认线程池
        # （max_workers≈min(32, cpu+4)），把 light:128 这类高上限静默节流到 ~32——
        # 于是 peak_concurrency 会虚高于真实 OS 并行度（"诚实分母"项目里尤其不能这样）。
        # 给足线程，分级信号量才是真正生效的上限，peak 才是诚实的在册并发高水位。
        pool = ThreadPoolExecutor(max_workers=max(1, sum(caps.values())))

        pq: asyncio.PriorityQueue = asyncio.PriorityQueue()
        for j in jobs:
            pq.put_nowait((j.priority, j.seq, j))

        async def run_one(job: Job) -> None:
            cls = job.resource_class
            sem = sems.get(cls) or sems.setdefault(cls, asyncio.Semaphore(8))
            async with sem:  # 分级并发上限
                async with lock:
                    active[cls] += 1
                    peak[cls] = max(peak[cls], active[cls])
                try:
                    res = await loop.run_in_executor(pool, run_job_with_retries, job, max_attempts)
                finally:
                    async with lock:
                        active[cls] -= 1
            async with lock:
                results.append(res)
                if store:
                    store.record(res)
                if on_result:
                    on_result(res)

        async def worker() -> None:
            while True:
                try:
                    _, _, job = pq.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await run_one(job)

        n_workers = max(1, sum(caps.values()))
        try:
            await asyncio.gather(*(worker() for _ in range(n_workers)))
        finally:
            pool.shutdown(wait=True)
        return results, dict(peak)


class ProcessExecutor:
    name = "process"

    def run(self, jobs, caps, max_attempts, store=None, on_result=None):
        from concurrent.futures import ProcessPoolExecutor, as_completed

        mode = os.environ.get("LOOM_OTEL", "off")
        # 每个资源类一个进程池，max_workers = 该类并发上限 → 结构性保证 peak ≤ cap
        pools: dict[str, Any] = {
            cls: ProcessPoolExecutor(max_workers=max(1, c), initializer=_proc_init, initargs=(mode,))
            for cls, c in caps.items()
        }
        results: list[JobResult] = []
        futmap = {}
        try:
            for j in jobs:
                pool = pools.get(j.resource_class)
                if pool is None:
                    pool = pools[j.resource_class] = ProcessPoolExecutor(
                        max_workers=8, initializer=_proc_init, initargs=(mode,))
                futmap[pool.submit(run_job_with_retries, j, max_attempts)] = j
            for fut in as_completed(list(futmap)):
                res = fut.result()
                results.append(res)
                if store:
                    store.record(res)
                if on_result:
                    on_result(res)
        finally:
            for p in pools.values():
                p.shutdown()

        cnt = Counter(j.resource_class for j in jobs)
        peak = {cls: min(cnt.get(cls, 0), caps.get(cls, 8)) for cls in set(list(caps) + list(cnt))}
        return results, peak


def _proc_init(otel_mode: str) -> None:
    """进程池 worker 初始化：在子进程里配置 OTel（trace 经 carrier 跨进程链接）。"""
    from loom.obs import setup_tracing

    setup_tracing(mode=otel_mode)


def make_executor(name: str) -> Executor:
    if name == "process":
        return ProcessExecutor()
    if name == "async":
        return AsyncExecutor()
    raise ValueError(f"未知 executor: {name}（可选 async / process）")
