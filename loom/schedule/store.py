"""SQLite RunStore —— 持久化每个 rollout 结果，支撑断点续跑与 dead-letter。

崩溃/中断后用同一 run_id 续跑：已 completed 的 task_id 直接跳过（幂等）。
耗尽重试的任务进 dead_letter 表，可追溯、不静默丢弃。
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
            CREATE TABLE IF NOT EXISTS dead_letter(
              run_id TEXT, task_id TEXT, attempts INTEGER, error TEXT,
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
            if res.status == "dead":
                self._conn.execute(
                    "INSERT OR REPLACE INTO dead_letter VALUES(?,?,?,?)",
                    (res.run_id, res.task_id, res.attempts, res.error or ""),
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
        with self._lock:
            cur = self._conn.execute(
                "SELECT task_id, attempts, error FROM dead_letter WHERE run_id=?", (run_id,))
            return [{"task_id": r[0], "attempts": r[1], "error": r[2]} for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
