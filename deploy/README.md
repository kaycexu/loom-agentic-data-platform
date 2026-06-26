# 部署 / 可观测 / 扩展

## 链路追踪（OpenTelemetry → Jaeger）

```bash
docker compose -f deploy/docker-compose.yaml up -d     # 起 Jaeger
pip install -e ".[obs]"
LOOM_OTEL=otlp loom demo                                # 或任意命令加 --otel otlp
# 打开 http://localhost:16686 ，service=loom
```

Span 树：`loom.schedule → loom.rollout → loom.step` 与 `loom.verify → loom.check`，
跨线程（AsyncExecutor）与跨进程（ProcessExecutor，经 OTLP + traceparent carrier）统一在一条 trace 下。
离线无 Jaeger 时用 `--otel console` 直接在 stdout 看 span 树。

## 执行后端（同一 Job 抽象，三种隔离层级）

| executor | 隔离 | 用途 |
|---|---|---|
| `async`（默认） | in-process 线程 + 独立 env 实例 | 单机高并发，OTel 全 span 树 |
| `process` | 真 OS 进程（每资源类一个进程池，size=并发上限） | 真隔离 + 多核 |
| K8s（seam） | 1 rollout = 1 Job/Pod | 横向扩展 |

```bash
loom scale --n 1000 --executor async                 # 单机
loom scale --n 1000 --executor process --light 32    # 真进程池
loom scale --n 1000 --store out/run.db               # 持久化（支持 --resume 断点续跑）
```

## K8s 横向扩展（seam）

调度器即 controller：每个 rollout 提交为一个 K8s Job，资源 request 按 `resource_class` 匹配。
样例 manifest 见 `deploy/k8s/*.yaml`，由以下命令生成：

```bash
loom k8s-manifest --n 3 --out deploy/k8s
```

Pod 容器入口即 `loom run-job --task-id <id> --policy <spec>`（OTel 经 `LOOM_OTEL=otlp` +
`OTEL_EXPORTER_OTLP_ENDPOINT` 把 trace 汇聚到 Jaeger）。本仓库不实际 apply，但 manifest 真实可读，
证明本地 async/process 后端与 K8s 后端共用同一个 `Job` 抽象与 `execute_job` 入口。

## 弹性

- 重试 + 指数退避（确定性 jitter）；耗尽进 **dead-letter**（`status='dead'`，可追溯，绝不静默丢弃）。
- SQLite run store：崩溃/中断后用同一 `run_id --resume` 跳过已完成任务，幂等续跑。
