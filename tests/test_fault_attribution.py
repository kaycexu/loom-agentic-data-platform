"""Fault attribution 端到端回归 —— 把"环境坏了"驱动穿过 runner→jobs→scheduler→curate。

证明核心不变式：reward 只在合法 rollout（SIGNAL）上计算，基建噪声（ENV_FAULT）
在结构上不可能进入数据集；且基建故障与"合法 reward=0 负样本"被严格区分对待。

各 subagent 已单测了局部（pool 的池逻辑、curate 的分流、scheduler 的会计）；
本文件补的是把真实 ENV_FAULT 从 env 一路驱动到交付的端到端链路。
"""

from __future__ import annotations

from loom.contracts import Outcome, SIGNAL_OUTCOMES
from loom.curate import curate
from loom.envs import EnvFault, SheetEmailEnv
from loom.rollout import run_rollout
from loom.schedule import schedule_tasks
from loom.schedule.jobs import Job, run_job_with_retries
from loom.schedule.store import RunStore
from loom.tasks import canonical_tasks, rubric_for


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class AlwaysFaultEnv:
    """reset 即抛 EnvFault——模拟浏览器/子进程起不来的基建故障。"""

    resource_profile = "light"

    def reset(self, seed):
        raise EnvFault("injected: simulated env crash on reset")

    def step(self, action):  # pragma: no cover - reset 先抛，到不了这
        raise EnvFault("dead")

    def get_state(self):  # pragma: no cover
        return {}

    def tools(self):  # pragma: no cover
        return []

    def close(self):
        pass


class MarkerFaultEnv:
    """按 seed 标记决定是否故障：有 _inject_fault 则 reset 抛 EnvFault，否则当作正常 SheetEmailEnv。

    用来在一次调度里混合"环境坏了"和"正常完成"两种 rollout。"""

    resource_profile = "light"

    def __init__(self):
        self._inner = SheetEmailEnv()

    def reset(self, seed):
        if seed and seed.get("_inject_fault"):
            raise EnvFault("injected: simulated browser crash on reset")
        return self._inner.reset(seed)

    def step(self, action):
        return self._inner.step(action)

    def get_state(self):
        return self._inner.get_state()

    def tools(self):
        return self._inner.tools()

    def close(self):
        try:
            self._inner.close()
        except Exception:
            pass


class FaultyPolicy:
    """act() 抛异常——模拟模型输出非法/崩溃（POLICY_ERROR，合法的策略失败信号）。"""

    name = "faulty"
    usage: dict = {}

    def reset(self, task, tools):
        pass

    def act(self, obs):
        raise ValueError("policy produced unparseable action")


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def test_env_fault_retried_then_quarantined_not_signal(monkeypatch):
    """ENV_FAULT → 幂等重试 → 耗尽进 quarantine；绝不计入信号、reward 占位 0。"""
    monkeypatch.setattr("loom.envs.make_env",
                        lambda env_type="browser", prefer_browser=False: AlwaysFaultEnv())
    task = canonical_tasks()[0]
    job = Job(run_id="q", task=task, policy_spec="mock:correct", resource_class="light")

    res = run_job_with_retries(job, max_attempts=3)

    assert res.outcome == Outcome.ENV_FAULT
    assert res.status == "quarantined"
    assert res.attempts == 3           # 基建故障被重试到耗尽
    assert res.report.passed is False
    assert res.report.total_reward == 0.0
    assert res.is_signal is False      # 关键：不是训练信号
    assert "injected" in (res.error or "")


def test_legit_reward_zero_is_completed_signal_not_fault():
    """合法 reward=0/低（模型真做错了）→ COMPLETED、是信号、不重试——与基建故障严格区分。"""
    task = canonical_tasks()[0]
    job = Job(run_id="neg", task=task, policy_spec="mock:missing_fill", resource_class="light")

    res = run_job_with_retries(job, max_attempts=3)

    assert res.outcome == Outcome.COMPLETED
    assert res.status == "completed"
    assert res.attempts == 1           # 合法失败不重试（不浪费在"模型真错了"上）
    assert res.report.passed is False  # 是负样本……
    assert res.is_signal is True       # ……但它是合法信号，可作负样本进 RL 数据


def test_policy_error_is_signal():
    """policy.act 崩 → POLICY_ERROR，归为合法信号（不重试）。"""
    task = canonical_tasks()[0]
    traj = run_rollout(task, SheetEmailEnv(), FaultyPolicy())

    assert traj.outcome == Outcome.POLICY_ERROR
    assert Outcome.POLICY_ERROR in SIGNAL_OUTCOMES  # 设计上：是信号、不重试
    assert "unparseable" in (traj.fault_detail or "")


def test_mixed_fault_and_signal_end_to_end(tmp_path, monkeypatch):
    """一次调度里 2 个环境故障 + 3 个正常完成：
    - 诚实分母正确（completed=3、env_fault_quarantined=2、pass_rate 分母只含 3）；
    - 故障不进 records；store 里 quarantined 可追溯；
    - curate 在混合记录上把故障排除出数据集，rollout_accounting 自洽。"""
    monkeypatch.setattr("loom.envs.make_env",
                        lambda env_type="browser", prefer_browser=False: MarkerFaultEnv())

    tasks = canonical_tasks()
    # 标记前两个任务的环境会崩
    for t in tasks[:2]:
        t.env_seed = {**t.env_seed, "_inject_fault": True}
    fault_ids = {tasks[0].task_id, tasks[1].task_id}
    tasks_by_id = {t.task_id: t for t in tasks}

    db = str(tmp_path / "mix.db")
    res = schedule_tasks(
        tasks, policy_for=lambda t, i: "mock:correct", class_for=lambda t, i: "light",
        run_id="mix", store_path=db, max_attempts=2)

    s = res.summary
    assert s["attempted"] == 5
    assert s["env_fault_quarantined"] == 2
    assert s["completed"] == 3
    assert s["pass_rate"] == 1.0                 # 分母只含 3 个合法信号，全过
    assert len(res.records) == 3                 # 故障不进下游
    assert all(t.task_id not in fault_ids for t, _, _ in res.records)

    # store 里 quarantined 可追溯、completed 只剩 3
    store = RunStore(db)
    assert len(store.quarantined("mix")) == 2
    assert len(store.completed_task_ids("mix")) == 3
    store.close()

    # 用「全量」job_results（含 2 个 ENV_FAULT 轨迹）喂 curate，验证它结构性排除故障
    records_full = [(tasks_by_id[r.task_id], r.trajectory, r.report) for r in res.job_results]
    manifest = curate(records_full, rubric_for, out_dir=str(tmp_path / "ds"),
                      policy_model="mock-oracle")
    acc = manifest.rollout_accounting
    assert acc["attempted"] == 5
    assert acc["by_outcome"]["env_fault"] == 2
    assert acc["excluded_faults"] == 2
    assert acc["signal"] == 3
    # sft.jsonl 里绝不含任何故障任务
    sft_lines = (tmp_path / "ds" / "sft.jsonl").read_text(encoding="utf-8").splitlines()
    assert sft_lines and all(fid not in line for line in sft_lines for fid in fault_ids)
