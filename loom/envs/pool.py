"""WarmEnvPool —— 重量环境（browser_heavy）的 warm 复用池 + 崩溃驱逐点。

起一个真实浏览器很贵（数百 ms–秒）。本池在 rollout 之间复用底层 chromium/Flask
进程（acquire 拿回空闲实例、runner 再 env.reset(seed) 做隔离），把吞吐拉起来；同时
把「fault attribution」落到资源层：一旦某实例曾触发 EnvFault（release(healthy=False)），
就驱逐销毁、绝不复用——坏实例不能污染下一条 rollout。

调用约定（与 schedule/jobs.py 写死的一致，本模块匹配之）：
    pool = get_pool(env_type, prefer_browser=True)
    env  = pool.acquire()          # 不带 seed；runner 自己 env.reset(seed) 做每-rollout 隔离
    ...                             # 跑 rollout
    pool.release(env, healthy=...)  # healthy=False（曾 EnvFault）→ 驱逐销毁

线程安全：AsyncExecutor 的多个线程共享同一进程内的池，所有共享态用一把 Lock 守护。
进程池后端下每个 worker 进程各有自己的单例，天然隔离——无需跨进程共享。
"""

from __future__ import annotations

import threading
from typing import Callable

from loom.envs import make_env
from loom.envs.base import Environment


def _env_alive(env: Environment) -> bool:
    """复用前的健康判定。

    真实 BrowserEnv 暴露 is_alive() 探针（浏览器 connected、Flask 子进程在跑、page 未关闭）；
    轻量 env（SheetEmailEnv 等）没有崩溃概念也无此方法 → 视为永远健康。
    探针本身抛错按「不活」处理（保守：宁可重建也不复用坏实例）。"""
    probe = getattr(env, "is_alive", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:
        return False


class WarmEnvPool:
    """进程内、线程安全的 warm 环境池。

    stats 至少含 created / reused / evicted / acquired / released（可观测）。
    """

    def __init__(
        self,
        factory: Callable[[], Environment],
        *,
        max_idle: int = 4,
    ) -> None:
        self._factory = factory
        self._max_idle = max_idle
        self._idle: list[Environment] = []
        self._lock = threading.Lock()
        self._closed = False
        self.stats: dict[str, int] = {
            "created": 0,
            "reused": 0,
            "evicted": 0,
            "acquired": 0,
            "released": 0,
        }

    # ----------------------------- 协议接口 ----------------------------- #
    def acquire(self) -> Environment:
        """取一个健康环境：优先复用空闲池里仍 alive 的实例，否则新建。

        不带 seed —— runner 会对返回的 env 自己调 reset(seed) 做每-rollout 隔离。
        复用前对实例探活，不活的就地 close 丢弃（计入 evicted），继续找下一个。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("pool already closed")
            while self._idle:
                env = self._idle.pop()
                if _env_alive(env):
                    self.stats["reused"] += 1
                    self.stats["acquired"] += 1
                    return env
                # 空闲期间坏掉的实例：驱逐，不复用。
                self._safe_close(env)
                self.stats["evicted"] += 1
            # 没有可复用的健康实例 → 新建。
            env = self._factory()
            self.stats["created"] += 1
            self.stats["acquired"] += 1
            return env

    def release(self, env: Environment, *, healthy: bool) -> None:
        """归还环境。

        healthy=False（曾触发 EnvFault）→ 驱逐：close 且不放回，evicted++。
        healthy=True → 放回空闲池；但若实例已不 alive，或空闲池已满（> max_idle），
        则 close 丢弃（避免无界堆积 / 复用坏实例）。"""
        with self._lock:
            self.stats["released"] += 1

            if not healthy:
                self._safe_close(env)
                self.stats["evicted"] += 1
                return

            # 健康归还，但池关了 / 实例其实已死 / 空闲已满 → 直接销毁。
            if self._closed or not _env_alive(env) or len(self._idle) >= self._max_idle:
                self._safe_close(env)
                self.stats["evicted"] += 1
                return

            self._idle.append(env)

    def close(self) -> None:
        """关闭池：销毁所有空闲实例，标记关闭（之后 acquire 会报错）。"""
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for env in idle:
            self._safe_close(env)

    # ----------------------------- 内部 ----------------------------- #
    @staticmethod
    def _safe_close(env: Environment) -> None:
        try:
            env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 进程内单例：按 (env_type, prefer_browser) 缓存，懒创建、线程安全。
# 进程池后端下每个 worker 进程各持有自己的 _POOLS——天然隔离，无需跨进程共享。
# --------------------------------------------------------------------------- #
_POOLS: dict[tuple[str, bool], WarmEnvPool] = {}
_POOLS_LOCK = threading.Lock()


def get_pool(
    env_type: str,
    *,
    prefer_browser: bool = True,
    max_idle: int = 4,
) -> WarmEnvPool:
    """取进程内单例池（按 (env_type, prefer_browser) 缓存）。

    默认 factory 为 make_env(env_type, prefer_browser=prefer_browser)；
    browser 类型在依赖缺失时 make_env 会降级到轻量 env，不影响主链路。"""
    key = (env_type, prefer_browser)
    pool = _POOLS.get(key)
    if pool is not None:
        return pool
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = WarmEnvPool(
                lambda: make_env(env_type, prefer_browser=prefer_browser),
                max_idle=max_idle,
            )
            _POOLS[key] = pool
        return pool
