"""WarmEnvPool 单测 —— 不依赖真实 Playwright，用可控 fake env 测池逻辑。

覆盖：warm 复用、崩溃驱逐、max_idle 上限、空闲期坏实例在 acquire 时被探活驱逐、
close() 回收、get_pool 进程内单例。
真实浏览器路径由 tests/test_browser_env.py 在依赖可用时单独验证。
"""

from __future__ import annotations

from typing import Any

from loom.envs.base import LIGHT, EnvFault, ToolSchema
from loom.envs.pool import WarmEnvPool, get_pool


class FakeEnv:
    """实现 Environment 协议的内存 env，行为可被测试指令控制。

    - is_alive()：默认 True；置 self.alive=False 模拟空闲期崩溃（供 acquire 探活驱逐）。
    - step()：若 self.raise_on_step，则抛 EnvFault 模拟基建故障。
    - closed：记录是否被 close（供断言驱逐/回收）。
    """

    resource_profile = LIGHT

    def __init__(self) -> None:
        self.closed = False
        self.alive = True
        self.raise_on_step = False
        self.reset_count = 0

    def reset(self, seed: dict[str, Any]) -> dict[str, Any]:
        self.reset_count += 1
        return {"last_result": "reset"}

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.raise_on_step:
            raise EnvFault("fake crash")
        return {"last_result": "ok"}

    def get_state(self) -> dict[str, Any]:
        return {"sheet": {"name": "Q2", "cells": {}, "rows": []},
                "email": {"id": "m1", "status": "unread"}}

    def tools(self) -> list[ToolSchema]:
        return [ToolSchema("noop", "no-op", {})]

    def is_alive(self) -> bool:
        return self.alive

    def close(self) -> None:
        self.closed = True


def _pool(max_idle: int = 4) -> tuple[WarmEnvPool, list[FakeEnv]]:
    """造一个用 FakeEnv 工厂的池，并返回已创建实例列表供断言。"""
    created: list[FakeEnv] = []

    def factory() -> FakeEnv:
        e = FakeEnv()
        created.append(e)
        return e

    return WarmEnvPool(factory, max_idle=max_idle), created


# --------------------------------------------------------------------------- #
def test_reuse_healthy_env():
    """acquire → release(healthy=True) → 再 acquire：拿到同一实例，reused++，created 不变。"""
    pool, created = _pool()

    e1 = pool.acquire()
    assert pool.stats["created"] == 1
    assert pool.stats["acquired"] == 1

    pool.release(e1, healthy=True)
    assert pool.stats["released"] == 1
    assert pool.stats["evicted"] == 0
    assert not e1.closed  # 健康归还不关闭

    e2 = pool.acquire()
    assert e2 is e1  # 复用同一实例
    assert pool.stats["reused"] == 1
    assert pool.stats["created"] == 1  # 没有新建
    assert len(created) == 1


def test_evict_unhealthy_env():
    """release(healthy=False)：实例被 close、evicted++，下次 acquire 新建（created++）。"""
    pool, created = _pool()

    e1 = pool.acquire()
    pool.release(e1, healthy=False)  # 曾触发 EnvFault → 驱逐
    assert e1.closed
    assert pool.stats["evicted"] == 1
    assert pool.stats["released"] == 1

    e2 = pool.acquire()
    assert e2 is not e1  # 不复用坏实例
    assert pool.stats["created"] == 2
    assert pool.stats["reused"] == 0
    assert len(created) == 2


def test_max_idle_cap():
    """连续 release 超过 max_idle 个健康 env，多余的被 close 丢弃。"""
    pool, created = _pool(max_idle=2)

    envs = [pool.acquire() for _ in range(4)]
    assert pool.stats["created"] == 4

    for e in envs:
        pool.release(e, healthy=True)

    # 只有 max_idle=2 个放回空闲，其余 2 个被 close 丢弃（计 evicted）。
    closed = [e for e in envs if e.closed]
    assert len(closed) == 2
    assert pool.stats["evicted"] == 2
    assert pool.stats["released"] == 4


def test_dead_idle_env_evicted_on_acquire():
    """空闲池里的实例在 acquire 前崩溃 → 被探活发现、close 驱逐，转而新建。"""
    pool, created = _pool()

    e1 = pool.acquire()
    pool.release(e1, healthy=True)  # 健康归还，进空闲池
    e1.alive = False  # 模拟空闲期间底层崩溃

    e2 = pool.acquire()
    assert e1.closed  # 坏的空闲实例被驱逐
    assert e2 is not e1
    assert pool.stats["evicted"] == 1
    assert pool.stats["created"] == 2  # 重新造了一个
    assert pool.stats["reused"] == 0


def test_unhealthy_at_release_even_if_flag_true():
    """healthy=True 但实例其实已死（探活失败）→ 仍驱逐，不放回污染下一条 rollout。"""
    pool, created = _pool()

    e1 = pool.acquire()
    e1.alive = False
    pool.release(e1, healthy=True)  # 标记健康，但探活说已死

    assert e1.closed
    assert pool.stats["evicted"] == 1
    assert len(pool._idle) == 0


def test_close_releases_all_idle():
    """close() 关闭所有空闲 env；之后 acquire 报错。"""
    pool, created = _pool()

    e1 = pool.acquire()
    e2 = pool.acquire()
    pool.release(e1, healthy=True)
    pool.release(e2, healthy=True)

    pool.close()
    assert e1.closed and e2.closed

    try:
        pool.acquire()
    except RuntimeError:
        pass
    else:
        raise AssertionError("close 后 acquire 应报错")


def test_release_after_close_evicts():
    """池关闭后归还（in-flight 的 env 跑完才回来）→ 直接 close，不复活池。"""
    pool, created = _pool()
    e1 = pool.acquire()
    pool.close()

    pool.release(e1, healthy=True)
    assert e1.closed
    assert pool.stats["evicted"] == 1


def test_get_pool_singleton():
    """同 key 两次调用返回同一对象；不同 key 返回不同对象。"""
    p1 = get_pool("light", prefer_browser=False)
    p2 = get_pool("light", prefer_browser=False)
    assert p1 is p2

    p3 = get_pool("light", prefer_browser=True)
    assert p3 is not p1  # 不同 prefer_browser → 不同池

    # 确保确实是个可用的池（acquire/close 跑得通；light → SheetEmailEnv，无依赖）。
    env = p1.acquire()
    assert hasattr(env, "reset")
    p1.release(env, healthy=True)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
