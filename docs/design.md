# Agentic 数据 + 环境生产平台 — 设计文档

> 代号：**Loom**（把验证过的 agentic 轨迹"织"成训练数据）
> 日期：2026-06-26
> 状态：设计已对齐，待实现

---

## 1. 背景与问题重构

客户（模型实验室）的原始需求很模糊：

> "提升模型在真实多步骤 agentic 任务上的能力——比如读懂一封邮件的要求，在一个应用/表格里操作几步、产出结果。给我这类任务的高质量训练环境和数据。"

**关键认知（决定整套设计的立场）：**

1. **交付给模型公司的核心是「数据本身」，其中最核心的是 Task + Rubric。** Trajectory 是 commodity——谁都能打各家模型的轨迹，不是壁垒。
2. **RL environment 不是产品，而是"数据生产后期做验证时用的环境"。** 环境服务于数据，不是反过来。
3. **真正的壁垒 = 怎么「定义」任务、怎么「蒸」好、怎么「验证」好。** 验证好 → 给客户模型在 RL 阶段更好的信号（PRM 式过程奖励）。
4. 我们**不做给客户的自定义 harness**；客户拿数据做 Agentic RL / 蒸馏（SFT）来训练模型本身。我们内部有一套数据生成 harness，仅服务于数据生产。

因此本平台不是"给客户一个在线 RL 环境"，而是**一条 agentic 数据生产线**：以 Task 设计 + Rubric + 多层 Verifier 为核心壁垒，环境是验证底座，规模 ≥1k 且并发隔离，最终交付**高质量、可验证、可复现的数据**。

---

## 2. 目标与非目标

### 目标
- 一套端到端可演示的数据生产线：任务/环境定义 → rollout → 多层验证 → 筛选 → 交付。
- **Verifier / Rubric 引擎**真实可跑、可组合、可解释、可复跑（核心壁垒）。
- 浏览器环境真实可用（最小 Web 应用 + Playwright），验证落在真实环境状态上。
- 设计上充分支持 **1k+ 任务、并发、rollout 隔离、资源感知调度、失败恢复**。
- 一个**度量验证器本身**的质量层（误收/误拒/泄露/judge 方差）——区别于"只是跑了 verifier"。
- 可视化 preview（看板/报告）+ 含架构图的 README。

### 非目标（YAGNI）
- 不做真正的分布式集群 / K8s 部署（用 asyncio 模拟，文档给出 K8s 映射）。
- 不真烧 1k 规模的 LLM token（真模型只跑小批 5~10 任务，1k 用 MockPolicy 模拟）。
- 不实现 API/文件/computer-use 等其它环境类型（只留接口，证明可扩展）。
- 不做训练侧（SFT/RL 训练循环）；只产出数据。
- 不做向量数据库 / 大规模存储；用本地文件（JSONL/SQLite）。

---

## 3. 目标用户与约束

- **客户**：国内大模型厂商（Tier1 frontier labs；Tier2 二线开始做模型的大厂，如美团、米哈游）。特点：很多连 Coding 数据都没做好，缺评测/验证 know-how。
- **规模约束**：系统设计需支持 ≥1000 条任务，并发跑，rollout 互相独立。
- **资源约束**：不同环境资源画像不同——browser/headless chrome 重内存，简单 env 轻量——需差异化调度。
- **可追溯**：每条 rollout 全程可 trace、可复现（影响最终交付物可信度）。

---

## 4. 架构总览

一条任务的生命周期（数据流主轴）：

```
TaskSpec (+ RubricSpec)
   │
   ▼  [Scheduler: 资源感知 / 隔离 / 并发 / 重试]
Environment.reset(seed) ──► Observation
   │
   ▼  [Rollout Runner: 内部数据生成 harness]
Policy.act ⇄ Env.step  (循环) ──► Trajectory + 终态 EnvState
   │
   ▼  [Verifier / Rubric 引擎 ★核心壁垒]
   ├─ 状态/结果检查（确定性，高可信）
   ├─ 过程/步骤检查（PRM 式 step reward）
   └─ LLM-judge 检查（模糊标准，结构化打分+理由）
   │  → RewardReport（总分 + 逐项拆解 + pass/fail）
   ▼  [Curator: 筛选 / 去重 / 配平 / 切分]
Dataset 交付物：
   ├─ SFT / 蒸馏数据
   ├─ RL 数据（task + env seed + reward fn + 离线 (traj, reward) + step reward）
   └─ Task+Rubric spec bundle（最有价值、可复跑、带版本）
   + Manifest（模型 / verifier 配置 / 版本，可复现）
   │
   ▼
Trace / Dashboard preview
   └─ 质量层指标（通过率、reward 分布、验证器可靠性）
```

横切关注点：**Trace/可观测**（每 rollout 一条 JSONL）、**Quality/Eval 元层**、**配置/版本**。

---

## 5. 核心数据契约（Pydantic v2）

契约层是整条流水线的"接口"，所有模块只依赖契约、不依赖彼此内部。

```python
# 任务定义（声明式）
class TaskSpec:
    task_id: str                 # 稳定唯一 id（幂等键）
    domain: str                  # 如 "email_to_sheet"
    difficulty: Literal["easy","medium","hard"]
    instruction: str             # 给 agent 的自然语言要求（邮件正文+目标）
    env_type: str                # "browser" | "api" | ...
    env_seed: dict               # 环境初始状态种子（决定可复现）
    allowed_tools: list[str]
    rubric_id: str               # 引用的 rubric
    max_steps: int = 20
    metadata: dict = {}

# 评分标准（可组合的 check 树）★壁垒
class RubricCheck:
    check_id: str
    kind: Literal["state","process","judge"]
    weight: float
    config: dict                 # 各类型的具体参数（断言/judge prompt 等）
    required: bool = False       # required 检查不过 → 整体 fail（安全/红线）

class RubricSpec:
    rubric_id: str
    version: str
    checks: list[RubricCheck]
    aggregation: Literal["weighted_sum","min","all_required"] = "weighted_sum"
    pass_threshold: float = 0.8

# rollout 产物（commodity）
class Step:
    index: int
    observation: dict
    thought: str | None
    action: dict                 # tool 调用 {name, args}
    tool_result: dict | None
    ts: float

class Trajectory:
    task_id: str
    attempt: int
    policy: str                  # 模型/policy 标识
    steps: list[Step]
    final_state: dict            # 终态 EnvState 快照（供 state 检查）
    status: Literal["completed","timeout","error","max_steps"]
    cost: dict                   # tokens / latency
    trace_id: str

# 验证结果
class CheckResult:
    check_id: str
    kind: str
    score: float                 # [0,1]
    passed: bool
    weight: float
    rationale: str               # 可解释（judge 给理由，state 给断言详情）

class RewardReport:
    task_id: str
    trace_id: str
    total_reward: float          # [0,1]
    passed: bool
    step_rewards: list[float]    # PRM 式过程奖励
    checks: list[CheckResult]
    verifier_version: str

# 交付物
class DatasetManifest:
    dataset_id: str
    created_at: str
    policy_model: str
    verifier_versions: dict
    counts: dict                 # total / kept / by_domain / by_difficulty
    reward_distribution: dict
    quality_metrics: dict        # 验证器可靠性指标
    provenance: dict             # 各组件版本、配置哈希
```

---

## 6. 模块设计（清晰边界）

每个模块：**做什么 / 接口 / 依赖 / 真跑 or 模拟**。

### 6.1 Task & Rubric 定义层（契约 + 壁垒）
- **做什么**：声明式定义任务与评分标准；提供任务生成器（模板/参数化，便于扩到 1k）。
- **接口**：`load_tasks() -> list[TaskSpec]`、`load_rubric(id) -> RubricSpec`、`generate_tasks(template, n) -> list[TaskSpec]`。
- **依赖**：契约层。
- **真跑**：5~10 个 `email_to_sheet` 真实任务 + rubric；外加一个任务生成器把同模板放大到 1k（给规模模拟用）。

### 6.2 Environment 抽象（验证底座）
- **做什么**：统一的环境接口；真实浏览器环境。
- **接口**：
  ```python
  class Environment(Protocol):
      resource_profile: ResourceProfile          # light / browser_heavy
      def reset(self, seed: dict) -> Observation: ...
      def step(self, action: dict) -> Observation: ...
      def get_state(self) -> dict: ...            # 供 state 检查
      def tools(self) -> list[ToolSchema]: ...
      def close(self) -> None: ...
  ```
- **真实实现 `BrowserEnv`**：
  - 一个最小本地 Web 应用（邮件收件箱 + 表格），Flask 起静态/动态页面。
  - Playwright 真实驱动（每实例独立 browser context = 廉价隔离）。
  - `get_state()` 读真实 DOM/应用状态 → 喂 verifier。
  - 资源画像 = `browser_heavy`。
- **接口-only**：`api`/`file`/`computer_use` 只声明，不实现（证明可扩展）。
- **真跑**：BrowserEnv 真跑。

### 6.3 Rollout Runner（内部数据生成 harness）
- **做什么**：跑 policy×env 的 tool-calling 循环，产出 Trajectory。
- **接口**：`run_rollout(task, env, policy) -> Trajectory`；`Policy.act(obs, scratchpad) -> action`。
- **Policy 实现**：`LLMPolicy`（真 Claude/GPT，tool-calling + scratchpad 维持长程记忆）、`MockPolicy`（脚本化/确定性，给规模模拟）。
- **安全**：高风险写操作走 pre-action hook 拦截（呼应候选人 KOC 项目的 pre-tool-use）。
- **真跑**：LLMPolicy 跑小批；MockPolicy 跑 1k。

### 6.4 Scheduler / 编排（隔离 + 并发 + 弹性，模拟实现）
- **做什么**：吃任务批，调度 rollout，保证隔离、并发上限、资源感知、失败恢复、可 trace。
- **接口**：`schedule(tasks, policy, concurrency_config) -> list[RewardReport]`。
- **并发模型（设计当真）**：
  - 每资源类一把**信号量**：`browser_heavy` 少并发（如 8），`light` 多并发（如 128）。
  - bounded queue 背压；每 rollout 独立 env 实例、**零共享可变状态**。
  - 超时 + 指数退避**重试**；每任务 attempt 上限；幂等 `task_id+attempt` → 写入去重。
  - checkpoint：已完成的 RewardReport 落盘 → 崩溃可恢复（断点续跑）。
- **K8s 映射（文档）**：scheduler=controller；每 rollout = Job/Pod，资源 request 匹配画像；browser env 作 in-pod/sidecar；结果写对象存储；asyncio worker 池是本地 stand-in。
- **真跑 or 模拟**：调度逻辑真实（asyncio + 信号量 + 重试 + checkpoint），**1k 规模用 MockPolicy 模拟**（快速跑完，展示吞吐/并发/trace）。

### 6.5 Curator / 数据集构建
- **做什么**：把验证过的 rollout 变成交付数据集。
- **接口**：`curate(reports, trajectories, policy) -> (Dataset, DatasetManifest)`。
- **逻辑**：按 reward 阈值筛选；去重（语义/结构）；按域/难度配平；train/val 切分。
- **导出三格式**：
  1. **SFT/蒸馏**：`{instruction, gold_trajectory}`（"模仿更好的数据"）。
  2. **RL**：`{task, env_seed, rubric_ref, offline (traj, reward), step_rewards}`。
  3. **Task+Rubric bundle**：带版本，客户可自行复跑/复验。
  - 附 `DatasetManifest`（provenance，可复现）。
- **真跑**：真跑。

### 6.6 Quality / Eval 元层（★面试官最在意）
- **做什么**：不只跑 verifier，还**度量 verifier 本身**的可靠性。
- **接口**：`evaluate_verifier(gold_set) -> QualityMetrics`。
- **指标**：
  - 在 **gold 子集**（人工标注的"正确轨迹"+ 故意"错误轨迹"）上算**误收率/误拒率**。
  - LLM-judge **多次方差/一致性**（跑 N 次看稳定性）。
  - **泄露率**：是否召回/通过了"一定不能通过"的样本（呼应候选人面试里的副样本/泄露思路）。
  - 数据集层：通过率（按域/难度）、reward 分布、去重统计。
- **真跑**：真跑（在小 gold 子集上）。

### 6.7 横切：Trace / Dashboard preview
- 每 rollout 一条 JSONL trace（task_id, attempt, status, reward 拆解, cost, duration, worker）。
- **preview**：静态 HTML 报告（或最小 FastAPI）：N 任务汇总、通过率、reward 分布、逐任务 trace 下钻、验证器可靠性面板。这是交付给面试官看的"产物预览"。

---

## 7. 并发与规模模型（设计当真）

- **隔离**：rollout 之间无共享可变状态；每个 env 实例独占（browser context / 进程 / pod）。
- **资源感知**：按 `resource_profile` 分级限流，避免 heavy env 打爆内存。
- **背压**：bounded task queue；生产者-消费者。
- **弹性**：超时、指数退避重试、attempt 上限、幂等去重、checkpoint 续跑。
- **可扩展到 1k+**：本地 asyncio 池 = 单机 stand-in；横向扩展时 1 rollout = 1 K8s Job/Pod，scheduler 变 controller，结果汇聚对象存储。
- **演示**：用 MockPolicy 把 1k 任务在分级信号量下跑完，输出吞吐/并发曲线 + trace，证明并发与隔离模型成立。

---

## 8. 真实实现 vs 模拟（边界明确）

| 组件 | 真跑 | 模拟 |
|---|---|---|
| Task/Rubric schema + 样例任务 | ✅ 5~10 个真实任务 + 生成器放大 | — |
| Verifier/Rubric 引擎 | ✅ 状态 + 过程 + LLM-judge 全功能 | — |
| BrowserEnv | ✅ 真 Web 应用 + Playwright | — |
| LLMPolicy | ✅ 真模型跑小批 | — |
| Curator + 导出 + manifest | ✅ | — |
| Quality/Eval 元层 | ✅ 小 gold 子集 | — |
| preview 看板 | ✅ | — |
| Scheduler 调度逻辑 | ✅ asyncio + 信号量 + 重试 + checkpoint | 1k 规模用 MockPolicy |
| 其它 env 类型 | — | 仅接口 |
| 分布式/K8s 部署 | — | 文档映射 |

---

## 9. 技术选型

| 关注点 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | 模型实验室/数据 infra 事实标准 |
| 数据契约 | Pydantic v2 | 强类型 schema + 校验 + 序列化 |
| 并发/调度 | asyncio | 单机模拟分布式调度，零额外依赖 |
| 浏览器环境 | Playwright | 真实驱动 + 廉价 context 隔离 |
| Web 应用 | Flask | 起最小邮件+表格应用 |
| LLM | Anthropic Claude（policy+judge），留 OpenAI 兼容开关 | 主力 + 可换 |
| 看板 | 静态 HTML + Jinja2 | 零部署、易交付 |
| CLI | typer | 一条命令跑全链路 |
| 测试 | pytest | 单测 verifier/curator/scheduler |

---

## 10. 仓库结构

```
copula/                         # 包名 loom
├── README.md                   # 含架构图（mermaid）、quickstart、设计取舍
├── docs/
│   └── design.md               # 本文档
├── pyproject.toml
├── loom/
│   ├── contracts/              # Pydantic 数据契约（第 5 节）
│   ├── tasks/                  # TaskSpec/RubricSpec 加载 + 生成器 + 样例任务
│   ├── envs/                   # Environment 抽象 + BrowserEnv + 最小 Web 应用
│   ├── rollout/                # Rollout runner + LLMPolicy / MockPolicy + hooks
│   ├── verify/                 # ★Verifier/Rubric 引擎（state/process/judge）
│   ├── schedule/               # Scheduler（信号量/重试/checkpoint）
│   ├── curate/                 # Curator + 三格式导出 + manifest
│   ├── quality/                # Quality/Eval 元层（gold 子集指标）
│   ├── trace/                  # JSONL trace + 看板生成
│   └── cli.py                  # typer 入口：run / scale / report
├── data/
│   ├── tasks/                  # 样例任务 + rubric（yaml/json）
│   └── gold/                   # gold 子集（正确+错误轨迹）
├── examples/                   # 端到端跑出来的产物 + 看板截图
└── tests/
```

---

## 11. Demo / Preview 计划

一条命令体现全链路：
1. `loom run --tasks data/tasks --policy claude --limit 8`：真模型跑 8 个浏览器任务 → 验证 → 导出数据集 + manifest。
2. `loom scale --n 1000 --policy mock`：MockPolicy 在分级信号量下跑 1k，输出并发/吞吐/trace。
3. `loom eval-verifier --gold data/gold`：输出验证器可靠性指标。
4. `loom report`：生成静态 HTML 看板（通过率、reward 分布、trace 下钻、质量面板）。

交付：GitHub 仓库 + README（架构图 + quickstart + 取舍说明）+ `examples/` 下的产物与看板截图。

---

## 12. 测试策略

- **契约**：Pydantic 校验 + 序列化往返。
- **Verifier**：对构造的"对/错"轨迹断言分数与 pass/fail；判定聚合逻辑。
- **Curator**：筛选/去重/配平/导出格式正确性。
- **Scheduler**：并发上限不被突破、重试触发、幂等去重、checkpoint 续跑。
- **Quality**：gold 子集上误收/误拒率计算正确。
- **冒烟**：CLI 端到端用 MockPolicy 跑通（CI 友好，不依赖真 LLM）。

---

## 13. 风险与开放问题

- **时间**：今晚午夜截止 → 严格按"真跑/模拟"边界控范围，BrowserEnv 与 LLMPolicy 优先保证 1 个 domain 跑通。
- **Playwright 安装/无头环境**：CI/无网时降级到 MockPolicy + mock env 状态，保证冒烟可跑。
- **LLM-judge 成本/方差**：judge 只在小批真跑；质量层量化方差。
- **导出格式与客户口径**：三格式为合理默认；真实对接时按客户 RL/SFT 框架定制（manifest 已预留 provenance）。

---

## 附：与面试信号的对应

| 面试官在意的点 | 本设计的承接 |
|---|---|
| 评测/验证体系（她最感兴趣） | 模块 6.5 Verifier 引擎 + 6.6 Quality 元层 |
| 评测指标（precision/recall/Hit@K/泄露率/多次稳定性） | Quality 元层指标 |
| 安全/可追溯（hooks/pre-tool-use/指令注入） | Rollout hooks + Trace + 红线 required 检查 |
| 并发/隔离/sandbox | Scheduler 并发模型 + env 实例隔离 |
| 长程任务稳定性（scratchpad/memory） | LLMPolicy scratchpad |
| 怎么定义/怎么蒸/怎么验证好 | Task/Rubric 定义 + Curator + Verifier 全链路 |
```
