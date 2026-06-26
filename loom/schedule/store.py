"""SQLite RunStore —— 持久化每个 rollout 结果，支撑断点续跑与故障可追溯。

崩溃/中断后用同一 run_id 续跑：已 completed 的 task_id 直接跳过（幂等）；
quarantined/dead 不算 completed，故续跑时会被重试，符合"基建/我方故障应重试"的预期。

三态故障追溯：耗尽重试的任务都进 fault_letter 表（status 列区分 'dead' vs 'quarantined'），
可追溯、不静默丢弃——env 故障（quarantined）与我方故障（dead）都看得见。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from loom.schedule.jobs import JobResult


class RunStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rollouts(
              run_id TEXT, task_id TEXT, status TEXT, passed INTEGER, reward REAL,
              attempts INTEGER, trace_id TEXT, json TEXT,
              PRIMARY KEY(run_id, task_id));
            CREATE TABLE IF NOT EXISTS fault_letter(
              run_id TEXT, task_id TEXT, status TEXT, attempts INTEGER, error TEXT,
              PRIMARY KEY(run_id, task_id));
            """
        )
        self._conn.commit()

    def record(self, res: JobResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO rollouts VALUES(?,?,?,?,?,?,?,?)",
                (res.run_id, res.task_id, res.status, int(res.report.passed),
                 res.report.total_reward, res.attempts, res.trajectory.trace_id,
                 res.model_dump_json()),
            )
            # 两类故障终态都落库可追溯：dead（HARNESS 耗尽）/ quarantined（ENV 耗尽）。
            if res.status in ("dead", "quarantined"):
                self._conn.execute(
                    "INSERT OR REPLACE INTO fault_letter VALUES(?,?,?,?,?)",
                    (res.run_id, res.task_id, res.status, res.attempts, res.error or ""),
                )
            self._conn.commit()

    def completed_task_ids(self, run_id: str) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT task_id FROM rollouts WHERE run_id=? AND status='completed'", (run_id,))
            return {r[0] for r in cur.fetchall()}

    def load_results(self, run_id: str) -> list[JobResult]:
        with self._lock:
            cur = self._conn.execute("SELECT json FROM rollouts WHERE run_id=?", (run_id,))
            return [JobResult.model_validate_json(r[0]) for r in cur.fetchall()]

    def dead_letters(self, run_id: str) -> list[dict]:
        """HARNESS 故障耗尽（status='dead'）的可追溯记录。"""
        return self._fault_letters(run_id, "dead")

    def quarantined(self, run_id: str) -> list[dict]:
        """ENV 故障耗尽（status='quarantined'）的可追溯记录——需运维介入、排除出数据集。"""
        return self._fault_letters(run_id, "quarantined")

    def _fault_letters(self, run_id: str, status: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT task_id, attempts, error FROM fault_letter WHERE run_id=? AND status=?",
                (run_id, status))
            return [{"task_id": r[0], "attempts": r[1], "error": r[2]} for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
