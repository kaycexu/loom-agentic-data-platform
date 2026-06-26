"""静态 HTML 看板 —— preview 交付物（重解释、轻美化）。

读取 run 目录的 trace.jsonl / quality.json / schedule.json / dataset/manifest.json，
渲染单文件 report.html：验证器可靠性面板 + 调度并发 + reward 分布 + 逐 rollout check 下钻。
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from loom.trace.store import RunDir

# autoescape=True：trace 来源字段（任务/邮件内容、policy、judge/tool 输出、check 理由）
# 注入 HTML 前自动转义，防止 trace 内容在看板里被当作 HTML/JS 执行（XSS）。
_TEMPLATE = Environment(autoescape=True).from_string(
    """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC",sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 .wrap{max-width:1080px;margin:0 auto;padding:24px}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:26px 0 10px;color:#9ad}
 .sub{color:#8a93a2;margin-bottom:16px}
 .chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
 .chip{background:#1a1f2b;border:1px solid #2a3240;border-radius:8px;padding:6px 10px}
 .chip b{color:#fff} .ok{color:#43d17a} .bad{color:#ff6b6b} .warn{color:#ffcf5c}
 table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #2a3240;padding:6px 8px;text-align:left}
 th{background:#161b25;color:#9ad} tr:nth-child(even) td{background:#141821}
 .badge{padding:1px 7px;border-radius:6px;font-size:12px;font-weight:600}
 .pass{background:#143524;color:#43d17a} .fail{background:#3a1722;color:#ff6b6b}
 .bars{display:flex;gap:2px;align-items:flex-end;height:22px}
 .bar{width:7px;background:#43d17a;border-radius:1px}
 .cm{display:grid;grid-template-columns:auto auto auto;gap:0;max-width:360px}
 .cm div{border:1px solid #2a3240;padding:8px 12px} .cm .hdr{background:#161b25;color:#9ad}
 details{background:#141821;border:1px solid #2a3240;border-radius:6px;margin:4px 0;padding:4px 8px}
 summary{cursor:pointer} code{color:#ffcf5c} .mut{color:#8a93a2}
 .hist{display:flex;gap:6px;align-items:flex-end;height:80px}
 .hist .col{display:flex;flex-direction:column;align-items:center;gap:4px;font-size:11px;color:#8a93a2}
 .hist .h{width:38px;background:#3a6df0;border-radius:2px 2px 0 0}
</style></head><body><div class="wrap">
<h1>{{ title }}</h1>
<div class="sub">{{ subtitle }}</div>
<div class="chips">
 <span class="chip">rollouts <b>{{ n }}</b></span>
 <span class="chip">pass <b class="ok">{{ n_pass }}</b> / fail <b class="bad">{{ n_fail }}</b></span>
 <span class="chip">本批 pass rate <b>{{ pass_rate }}</b></span>
 <span class="chip">policies <b>{{ policies|join(', ') }}</b></span>
</div>

{% if quality %}
<h2>① 验证器可靠性（核心壁垒：验证好才有好信号）</h2>
<div class="chips">
 <span class="chip">误收率 FA <b class="{{ 'ok' if quality.false_accept_rate==0 else 'bad' }}">{{ quality.false_accept_rate }}</b></span>
 <span class="chip">误拒率 FR <b class="{{ 'ok' if quality.false_reject_rate==0 else 'warn' }}">{{ quality.false_reject_rate }}</b></span>
 <span class="chip">红线泄露 <b class="{{ 'ok' if quality.leakage_count==0 else 'bad' }}">{{ quality.leakage_count }}</b>{% if quality.leakage_count==0 %} ✓{% endif %}</span>
 <span class="chip">预期失败命中 <b>{{ quality.expected_failed_hit_rate }}</b></span>
 <span class="chip">accuracy <b>{{ quality.accuracy }}</b></span>
</div>
<div class="cm">
 <div class="hdr"></div><div class="hdr">verifier: pass</div><div class="hdr">verifier: fail</div>
 <div class="hdr">应通过</div><div class="ok">TP {{ quality.confusion_matrix.tp }}</div><div class="warn">FN {{ quality.confusion_matrix.fn }}</div>
 <div class="hdr">应拒绝</div><div class="bad">FP {{ quality.confusion_matrix.fp }}</div><div class="ok">TN {{ quality.confusion_matrix.tn }}</div>
</div>
{% if judge_variance and judge_variance.judge_evaluated %}
<p class="mut">LLM-judge 稳定性（{{ judge_variance.runs }} 次）：mean {{ judge_variance.mean }}，stdev {{ judge_variance.stdev }}</p>
{% elif judge_variance %}<p class="mut">LLM-judge：{{ judge_variance.note }}</p>{% endif %}
{% endif %}

{% if schedule %}
<h2>② 调度 / 并发（资源感知 + 隔离，1k 模拟）</h2>
<div class="chips">
 <span class="chip">tasks <b>{{ schedule.total }}</b></span>
 <span class="chip">wall <b>{{ schedule.wall_clock_s }}s</b></span>
 <span class="chip">吞吐 <b>{{ schedule.throughput_per_s }}/s</b></span>
 <span class="chip">规模批 pass rate <b>{{ schedule.pass_rate }}</b><span class="mut"> （分母仅含合法信号 {{ schedule.completed }}）</span></span>
 <span class="chip">重试 <b>{{ schedule.retried }}</b></span>
</div>
<div class="chips">
 <span class="chip">合法信号 <b>{{ schedule.signal_count }}</b></span>
 <span class="chip">env 故障隔离 <b class="{{ 'ok' if schedule.env_fault_quarantined==0 else 'warn' }}">{{ schedule.env_fault_quarantined }}</b></span>
 <span class="chip">dead-letter <b class="{{ 'ok' if schedule.dead_letter==0 else 'warn' }}">{{ schedule.dead_letter }}</b></span>
 <span class="chip">rollout 执行次数(含重试) <b>{{ schedule.rollout_attempts }}</b></span>
</div>
<table><tr><th>资源类</th><th>并发上限</th><th>峰值并发</th><th>任务数</th></tr>
{% for cls, cap in schedule.configured_caps.items() %}
 <tr><td>{{ cls }}</td><td>{{ cap }}</td>
 <td class="{{ 'ok' if schedule.peak_concurrency.get(cls,0)<=cap else 'bad' }}">{{ schedule.peak_concurrency.get(cls,0) }}</td>
 <td>{{ schedule.by_resource_class.get(cls,0) }}</td></tr>
{% endfor %}</table>
<p class="mut">峰值并发 = 实测在册并发高水位（专用线程池保证分级上限真正可达，非默认线程池节流后的虚高值）。诚实分母：基建故障（env 隔离 / dead-letter）不计入 pass rate 分母、结构性排除出数据集。</p>
{% endif %}

{% if manifest %}
<h2>③ 数据集交付（筛选 / 去重 / 配平）</h2>
<div class="chips">
 <span class="chip">输入 <b>{{ manifest.counts.total_in }}</b></span>
 <span class="chip">保留 <b class="ok">{{ manifest.counts.kept }}</b></span>
 <span class="chip">低质丢弃 <b>{{ manifest.counts.dropped_low_quality }}</b></span>
 <span class="chip">去重丢弃 <b>{{ manifest.counts.dropped_duplicate }}</b></span>
 <span class="chip">policy <b>{{ manifest.policy_model }}</b></span>
</div>
<p class="mut">交付格式：SFT(sft.jsonl) · RL(rl.jsonl) · Task+Rubric bundle/ · manifest.json（provenance 可复现）。难度分布：{{ manifest.counts.by_difficulty }}</p>
<div class="hist">
{% for k, v in manifest.reward_distribution.items() %}
 <div class="col"><div class="h" style="height:{{ (v / max_hist * 70)|round(0,'floor')|int }}px"></div>{{ v }}<span>{{ k }}</span></div>
{% endfor %}</div>
{% endif %}

<h2>④ 逐 rollout（点开看每个 check 的解释 + step reward）</h2>
<table><tr><th>task</th><th>policy</th><th>判定</th><th>reward</th><th>step rewards</th><th>失败 checks</th></tr>
{% for r in rows %}
<tr>
 <td>{{ r.task_id }}<div class="mut">{{ r.difficulty }}</div></td>
 <td>{{ r.policy }}</td>
 <td>{% if r.passed %}<span class="badge pass">PASS</span>{% else %}<span class="badge fail">FAIL</span>{% endif %}</td>
 <td>{{ r.reward }}</td>
 <td><div class="bars">{% for sr in r.step_rewards %}<div class="bar" style="height:{{ (sr*20)|round(0,'floor')|int + 2 }}px;background:{{ '#43d17a' if sr>=0.99 else '#ff6b6b' }}"></div>{% endfor %}</div></td>
 <td>{% for c in r.checks if not c.passed and not c.skipped %}<code>{{ c.check_id }}</code>{% if c.required %}<span class="bad">*</span>{% endif %} {% endfor %}</td>
</tr>
<tr><td colspan="6"><details><summary class="mut">checks 明细（{{ r.actions|join(' → ') }}）</summary>
 <table><tr><th>check</th><th>类型</th><th>红线</th><th>score</th><th>判定</th><th>理由</th></tr>
 {% for c in r.checks %}<tr>
  <td><code>{{ c.check_id }}</code></td><td>{{ c.kind }}/{{ c.scope }}</td>
  <td>{{ '是' if c.required else '' }}</td><td>{{ c.score }}</td>
  <td>{% if c.skipped %}<span class="mut">skip</span>{% elif c.passed %}<span class="ok">✓</span>{% else %}<span class="bad">✗</span>{% endif %}</td>
  <td class="mut">{{ c.rationale }}</td></tr>{% endfor %}
 </table></details></td></tr>
{% endfor %}
</table>
<p class="mut" style="margin-top:24px">Loom · agentic 数据 + 环境生产平台 · 数据是产品，环境是验证底座，Task+Rubric+验证是壁垒。</p>
</div></body></html>"""
)


def render_report(run: RunDir, title: str = "Loom — agentic 数据生产 run", subtitle: str = "") -> Path:
    trace = run.load_trace()
    quality = run.load_json("quality.json")
    schedule = run.load_json("schedule.json")
    manifest = run.load_json("dataset/manifest.json")
    judge_var = run.load_json("judge_variance.json")

    n = len(trace)
    n_pass = sum(1 for r in trace if r.get("passed"))
    policies = sorted({r.get("policy") for r in trace if r.get("policy")})
    max_hist = 1
    if manifest and manifest.get("reward_distribution"):
        max_hist = max(max(manifest["reward_distribution"].values()), 1)

    html = _TEMPLATE.render(
        title=title,
        subtitle=subtitle or "数据流：Task/Rubric → Env → Rollout → Verify → Curate → Dataset",
        n=n, n_pass=n_pass, n_fail=n - n_pass,
        pass_rate=round(n_pass / n, 3) if n else 0.0,
        policies=policies,
        rows=trace, quality=quality, schedule=schedule, manifest=manifest,
        judge_variance=judge_var, max_hist=max_hist,
    )
    out = run.path("report.html")
    out.write_text(html, encoding="utf-8")
    return out
