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

## 弹性与 fault attribution

每条 rollout 先归因到一个 `Outcome`，再决定是否重试、耗尽后进哪个"垃圾桶"。**两个垃圾桶含义不同，运维动作也不同**：

- **quarantine（隔离区）= `ENV_FAULT` 耗尽**：环境/基建故障（浏览器 crash/hang、子进程退出、`navigate` 超时）反复重试仍失败。运维含义：**怀疑环境镜像/资源**——查 Playwright 版本、内存上限、宿主机负载、warm 池实例健康。重跑这批任务往往就能恢复（换台机器/扩容）。
- **dead-letter（死信）= `HARNESS_FAULT` 耗尽**：我方代码/配置故障（契约违反、序列化失败、bug）。运维含义：**这是我们的 bug**——光重跑不会好，要看堆栈、修代码/配置再重放。
- 两者都**绝不静默丢弃**：可追溯、记入交付 manifest 的 `rollout_accounting` 诚实分母，与**合法 `reward=0` 负样本**严格区分（后者是数据集想要的，前者必须排除）。
- 重试 + 指数退避（确定性 jitter）；一个 `EnvFault` 同时驱逐崩溃的池实例 + 换新实例幂等重试。
- SQLite run store：崩溃/中断后用同一 `run_id --resume` 跳过已完成任务，幂等续跑。

> 一句话区分：**quarantine 是"环境的锅"（怀疑基建、可重跑恢复），dead-letter 是"我们的锅"（修代码再重放）。** 把二者混在一个失败队列里，是大规模跑环境时最常见的运维盲区。完整归因模型见 [`../docs/design.md` §8](../docs/design.md)。
