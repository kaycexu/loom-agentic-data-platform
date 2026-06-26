"""K8s 执行 seam —— 把 Job 渲染成 "1 rollout = 1 K8s Job/Pod" 的 manifest。

横向扩展时调度器即 controller：每个 rollout 提交为一个 K8s Job，资源 request 按
resource_class 匹配（browser_heavy 要更多内存），结果写对象存储。这里不实际 apply，
而是生成真实可读的 manifest，证明本地 asyncio/进程池后端与 K8s 后端是同一个 Job 抽象。
"""

from __future__ import annotations

from pathlib import Path

from loom.schedule.jobs import Job

# 资源 request 按资源画像分级（与本地分级信号量一一对应）
RESOURCE_REQUESTS = {
    "light": {"cpu": "250m", "memory": "256Mi"},
    "browser_heavy": {"cpu": "1", "memory": "2Gi"},
}


def render_job_manifest(job: Job, image: str = "loom:latest") -> str:
    req = RESOURCE_REQUESTS.get(job.resource_class, RESOURCE_REQUESTS["light"])
    safe_id = job.task.task_id.replace("/", "-").replace("_", "-").lower()
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: loom-rollout-{safe_id}
  labels:
    app: loom
    loom/resource-class: {job.resource_class}
    loom/run-id: "{job.run_id}"
spec:
  backoffLimit: 2          # 与 max_attempts 对应：失败重试，耗尽进 dead-letter
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: rollout
          image: {image}
          args: ["run-job", "--task-id", "{job.task.task_id}", "--policy", "{job.policy_spec}"]
          env:
            - name: LOOM_OTEL
              value: "otlp"
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://jaeger-collector:4318"   # trace 汇聚到 Jaeger
          resources:
            requests: {{cpu: "{req['cpu']}", memory: "{req['memory']}"}}
            limits: {{cpu: "{req['cpu']}", memory: "{req['memory']}"}}
"""


def render_manifests(jobs: list[Job], out_dir: str | Path, limit: int = 3, image: str = "loom:latest") -> dict:
    """写出前 limit 个 Job 的 manifest 作为样例（不实际生成 1000 份）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for job in jobs[:limit]:
        p = out / f"job-{job.task.task_id.replace('/', '-')}.yaml"
        p.write_text(render_job_manifest(job, image=image), encoding="utf-8")
        written.append(str(p))
    return {"would_generate": len(jobs), "wrote_samples": written, "dir": str(out)}
