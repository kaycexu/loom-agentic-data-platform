"""Verifier 引擎 — 把 rubric 的多层 check 聚合成 RewardReport。

聚合规则（docs/design.md §5/§6.6）：
- total_reward = Σ(weight·score) / Σweight，仅统计未 skipped 的 check。
- 红线门控：任一 required check 不过 → passed=False（无论 total 多高）。
  这保证红线负样本"绝不被误判 pass"（leakage = 0 的机制基础）。
- passed = 所有 required 通过 且 total_reward >= pass_threshold。
- step_rewards：由 scope=step 的 check 的逐步信号按步求均值派生（PRM 式）；
  没有 step 级信号的步默认 1.0（无负向信号）。
"""

from __future__ import annotations

from loom.contracts import (
    CheckResult,
    RewardReport,
    RubricSpec,
    TaskSpec,
    Trajectory,
)
from loom.obs import span
from loom.verify.checks import check_process, check_state
from loom.verify.judge import JudgeClient, StubJudge

VERIFIER_VERSION = "v1"


class Verifier:
    def __init__(self, judge: JudgeClient | None = None):
        self.judge = judge or StubJudge()

    def verify(self, task: TaskSpec, rubric: RubricSpec, traj: Trajectory) -> RewardReport:
        results: list[CheckResult] = []
        step_signals: list[tuple[int, float]] = []  # (step_index, score)

        with span("loom.verify", **{"loom.task_id": task.task_id, "loom.rubric": rubric.rubric_id}) as vspan:
            for rc in rubric.checks:
                kind = rc.spec.kind
                with span("loom.check", **{"loom.check_id": rc.check_id, "loom.kind": kind,
                                           "loom.required": rc.required}) as cspan:
                    if kind == "state":
                        cr, ps = check_state(rc, traj)
                    elif kind == "process":
                        cr, ps = check_process(rc, traj)
                    elif kind == "judge":
                        cr = self._judge(rc, task, traj)
                        ps = []
                    else:  # pragma: no cover
                        raise ValueError(f"未知 check kind: {kind}")
                    if cspan is not None:
                        cspan.set_attribute("loom.passed", cr.passed)
                        cspan.set_attribute("loom.score", cr.score)
                        cspan.set_attribute("loom.skipped", cr.skipped)
                results.append(cr)
                step_signals.extend(ps)

            scored = [r for r in results if not r.skipped]
            wsum = sum(r.weight for r in scored) or 1.0
            total = sum(r.weight * r.score for r in scored) / wsum

            # fail-closed：required check 即使被 skip（judge 无 LLM/出错）也算 fail。
            # 绝不能让"没真正跑的强制检查"放行——这是守住 leakage=0 / 红线边界的关键。
            # （skipped 的 CheckResult.passed 已为 False，此处不再排除 skipped。）
            required_ok = all(r.passed for r in results if r.required)
            passed = required_ok and (total >= rubric.pass_threshold)
            step_rewards = self._step_rewards(traj, step_signals)

            if vspan is not None:
                vspan.set_attribute("loom.total_reward", round(total, 4))
                vspan.set_attribute("loom.passed", passed)

            return RewardReport(
                task_id=task.task_id,
                trace_id=traj.trace_id,
                total_reward=round(total, 4),
                passed=passed,
                step_rewards=step_rewards,
                checks=results,
                verifier_version=VERIFIER_VERSION,
                policy=traj.policy,
            )

    def _judge(self, rc, task: TaskSpec, traj: Trajectory) -> CheckResult:
        spec = rc.spec
        score, rationale = self.judge.score(spec.rubric_text, spec.scale, task.instruction, traj)
        skipped = score is None
        return CheckResult(
            check_id=rc.check_id,
            kind="judge",
            score=0.0 if skipped else round(float(score), 4),
            passed=(not skipped) and float(score) >= 0.6,
            weight=rc.weight,
            scope=spec.scope,
            required=rc.required,
            skipped=skipped,
            rationale=rationale,
        )

    @staticmethod
    def _step_rewards(traj: Trajectory, signals: list[tuple[int, float]]) -> list[float]:
        rewards: list[float] = []
        for i in range(len(traj.steps)):
            scs = [s for (idx, s) in signals if idx == i]
            rewards.append(round(sum(scs) / len(scs), 3) if scs else 1.0)
        return rewards
