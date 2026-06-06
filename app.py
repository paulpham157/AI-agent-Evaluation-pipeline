#!/usr/bin/env python3
"""
AI Agent Evaluation Pipeline — Gradio MVP
==========================================
Evaluate AI agents at 3 hierarchical levels, inspired by
Amazon Bedrock AgentCore Evaluations.

  📦 Session  — Did the agent achieve the user's goal?
  🔄 Trace    — Per-turn quality (11 evaluators)
  🔧 Span     — Per tool-call accuracy (2 evaluators)

Run locally : python app.py
HuggingFace : app_file = app.py  (Gradio SDK)
"""

import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable whether run from repo root or HF Spaces
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import gradio as gr

# HF ZeroGPU Spaces require at least one @spaces.GPU-decorated function
# to be detected at module load. The actual evaluation and dataset
# generation work in this app uses the cloud InferenceClient and runs
# without local GPU compute; the placeholder below exists only to
# satisfy the runtime's static check. `spaces` is pre-installed on
# ZeroGPU hardware; we guard the import so the app still loads if it
# is missing (e.g. local CPU dev).
try:
    import spaces as _spaces
except ImportError:
    class _spaces_stub:
        @staticmethod
        def GPU(fn, duration: int = 60):
            return fn
    _spaces = _spaces_stub()


@_spaces.GPU
def _zero_gpu_healthcheck() -> dict:
    """Placeholder GPU function detected by the ZeroGPU runtime."""
    try:
        import torch
        return {"cuda_available": bool(torch.cuda.is_available())}
    except ImportError:
        return {"cuda_available": False, "note": "torch not installed"}


from scripts.generate_golden_dataset import (
    DATASET_REPO,
    DOMAIN_LABELS,
    GENERATOR_MODEL,
    SCENARIOS,
    build_templates,
    call_model,
    make_prompt,
    parse_output,
    upload_to_hf,
    validate,
)
from src.evaluators import (
    ALL_EVALUATORS,
    DEFAULT_TRACE_EVALS,
    SESSION_EVALUATORS,
    SPAN_EVALUATORS,
    TRACE_EVALUATORS,
)
from src.llm_judge import LLMJudge
from src.models import EvalLevel, EvalMode, GroundTruth
from src.parser import format_trace_tree, parse_trace
from src.reliability import compute_reliability
from src.runner import EvalRunner
from src.visualizer import create_bar_chart, create_radar_chart, create_trace_timeline

# ─── Load demo traces ───────────────────────────────────────────────────────

_DEMOS = _ROOT / "demos"


def _load_demo(name: str) -> str:
    p = _DEMOS / f"{name}.json"
    return p.read_text(encoding="utf-8") if p.exists() else "{}"


DEMO_SIMPLE_QA = _load_demo("simple_qa")
DEMO_TOOL_CALLING = _load_demo("tool_calling")
DEMO_MULTI_TURN = _load_demo("multi_turn")

# ─── UI helpers ─────────────────────────────────────────────────────────────

_LEVEL_COLOR = {
    EvalLevel.SESSION: "#9B59B6",
    EvalLevel.TRACE: "#3498DB",
    EvalLevel.SPAN: "#27AE60",
}

_LEVEL_ICON = {
    EvalLevel.SESSION: "📦",
    EvalLevel.TRACE: "🔄",
    EvalLevel.SPAN: "🔧",
}


def _bar_color(score: float) -> str:
    if score >= 0.8:
        return "#4CAF50"
    elif score >= 0.6:
        return "#FF9800"
    return "#F44336"


def _bg_color(score: float) -> str:
    if score >= 0.8:
        return "rgba(76,175,80,0.12)"
    elif score >= 0.6:
        return "rgba(255,152,0,0.12)"
    return "rgba(244,67,54,0.12)"


def render_score_card(score) -> str:
    color = _bar_color(score.score)
    bg = _bg_color(score.score)
    badge_color = _LEVEL_COLOR.get(score.level, "#888")
    level_icon = _LEVEL_ICON.get(score.level, "")

    return f"""
<div style="background:{bg};border-radius:8px;padding:12px 15px;margin:5px 0;
            border-left:4px solid {color};border:1px solid rgba(255,255,255,0.07);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="background:{badge_color};color:white;padding:2px 7px;border-radius:4px;
                   font-size:10px;font-weight:700;letter-spacing:0.5px;">{level_icon} {score.level.value}</span>
      <span style="color:#eee;font-weight:600;font-size:13px;">{score.evaluator_display}</span>
    </div>
    <span style="background:{color};color:white;padding:3px 10px;border-radius:10px;
                 font-size:13px;font-weight:700;">{score.score_pct}%</span>
  </div>
  <div style="background:rgba(255,255,255,0.08);border-radius:3px;height:4px;margin-bottom:8px;">
    <div style="background:{color};height:4px;border-radius:3px;width:{score.score_pct}%;"></div>
  </div>
  <div style="color:rgba(210,210,210,0.85);font-size:11.5px;line-height:1.55;">
    <span style="color:rgba(150,150,150,0.7);font-size:10px;">
      {score.target_label} &nbsp;·&nbsp; {score.mode.value} mode
    </span><br>
    {score.explanation}
  </div>
</div>"""


def render_overall_banner(report) -> str:
    s = report.overall_score
    color = _bar_color(s)
    passed = sum(1 for x in report.scores if x.passed)
    total = len(report.scores)
    status = "PASS ✅" if s >= 0.6 else "NEEDS REVIEW ⚠️"

    # Level breakdown
    sess_avg = (
        sum(x.score for x in report.session_scores) / len(report.session_scores)
        if report.session_scores
        else None
    )
    trace_avg = (
        sum(x.score for x in report.trace_scores) / len(report.trace_scores)
        if report.trace_scores
        else None
    )
    span_avg = (
        sum(x.score for x in report.span_scores) / len(report.span_scores)
        if report.span_scores
        else None
    )

    def level_chip(label, avg, icon, level):
        if avg is None:
            return ""
        c = _bar_color(avg)
        bc = _LEVEL_COLOR.get(level, "#888")
        return (
            f'<div style="text-align:center;padding:8px 14px;background:rgba(255,255,255,0.06);'
            f'border-radius:8px;border:1px solid {bc}33;">'
            f'<div style="font-size:10px;color:{bc};font-weight:700;margin-bottom:3px;">{icon} {label}</div>'
            f'<div style="font-size:20px;font-weight:800;color:{c};">{avg:.0%}</div>'
            f"</div>"
        )

    chips = " ".join(
        [
            level_chip("SESSION", sess_avg, "📦", EvalLevel.SESSION),
            level_chip("TRACE", trace_avg, "🔄", EvalLevel.TRACE),
            level_chip("SPAN", span_avg, "🔧", EvalLevel.SPAN),
        ]
    )

    return f"""
<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            border-radius:12px;padding:20px 24px;margin:4px 0;
            border:1px solid rgba(255,255,255,0.1);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="color:rgba(180,180,180,0.8);font-size:11px;letter-spacing:1px;margin-bottom:4px;">OVERALL SCORE</div>
      <div style="font-size:42px;font-weight:800;color:{color};line-height:1;">{s:.0%}</div>
      <div style="color:rgba(180,180,180,0.7);font-size:12px;margin-top:6px;">
        {passed}/{total} evaluators passed &nbsp;·&nbsp;
        {len(report.session.traces)} turn(s) &nbsp;·&nbsp;
        {report.elapsed_seconds:.2f}s &nbsp;·&nbsp;
        {report.eval_mode.value} mode
      </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
      <div style="font-size:22px;font-weight:700;color:{color};">{status}</div>
      <div style="display:flex;gap:8px;">{chips}</div>
    </div>
  </div>
  <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;margin-top:16px;">
    <div style="background:{color};height:6px;border-radius:4px;width:{int(s * 100)}%;
                transition:width 0.5s ease;"></div>
  </div>
</div>"""


def parse_and_preview(trace_json: str) -> str:
    if not trace_json or not trace_json.strip():
        return "*Paste or load a JSON trace above to see a preview.*"
    try:
        session = parse_trace(trace_json)
        return format_trace_tree(session)
    except Exception as e:
        return f"❌ **Parse error:** `{e}`\n\nCheck that your JSON is valid and contains `user_goal` + `traces`."


# ─── Dataset generation function ───────────────────────────────────────────────


def run_dataset_generation(domains: list, n_per_domain: int, hf_token: str):
    """Gradio generator: yields (status_html, log_text) as generation progresses."""
    import json
    import time
    from pathlib import Path

    from huggingface_hub import InferenceClient

    output_path = _ROOT / "dataset" / "golden_dataset.jsonl"
    output_path.parent.mkdir(exist_ok=True)

    if not domains:
        yield (
            "<div style='color:#FF9800;padding:16px;'>⚠️ Select at least one domain.</div>",
            "",
        )
        return

    templates = build_templates(domains, int(n_per_domain))
    total = len(templates)
    log_lines = []

    def status_html(done, failed, total):
        pct = int(done / total * 100) if total else 0
        color = "#4CAF50" if failed == 0 else "#FF9800"
        return (
            f"<div style='padding:14px;background:rgba(255,255,255,0.05);border-radius:8px;'>"
            f"<div style='font-size:13px;color:#aaa;margin-bottom:6px;'>"
            f"✅ {done} done &nbsp;·&nbsp; ✗ {failed} failed &nbsp;·&nbsp; {total} total</div>"
            f"<div style='background:rgba(255,255,255,0.1);border-radius:3px;height:6px;'>"
            f"<div style='background:{color};height:6px;border-radius:3px;width:{pct}%;'></div>"
            f"</div></div>"
        )

    # Load already-generated IDs
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                existing_ids.add(json.loads(line)["id"])

    pending = [t for t in templates if t["id"] not in existing_ids]
    already_done = len(existing_ids)

    if not pending:
        yield status_html(already_done, 0, total), "All records already generated."
        return

    log_lines.append(f"Model: {GENERATOR_MODEL}")
    log_lines.append(f"Total: {total} records  |  Pending: {len(pending)}")
    log_lines.append("=" * 45)
    yield status_html(already_done, 0, total), "\n".join(log_lines)

    token = hf_token.strip() or None
    client = InferenceClient(model=GENERATOR_MODEL, token=token)

    done, failed = already_done, 0
    with open(output_path, "a", encoding="utf-8") as f:
        for t in pending:
            log_lines.append(f"⏳ {t['id']} ({t['domain']}/{t['difficulty']})...")
            yield status_html(done, failed, total), "\n".join(log_lines)

            rec = call_model(client, t)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                log_lines[-1] = f"✅ {t['id']} ({t['domain']}/{t['difficulty']})"
            else:
                failed += 1
                log_lines[-1] = f"✗  {t['id']} — parse failed"

            yield status_html(done, failed, total), "\n".join(log_lines)
            time.sleep(0.3)

    # Upload to HF
    log_lines.append("")
    log_lines.append("📤 Uploading to HuggingFace dataset repo...")
    yield status_html(done, failed, total), "\n".join(log_lines)
    try:
        upload_to_hf(output_path, hf_token=token)
        log_lines.append(f"✓ Uploaded → {DATASET_REPO}")
    except Exception as e:
        log_lines.append(f"✗ Upload failed: {e}")

    final_html = (
        f"<div style='padding:14px;background:rgba(76,175,80,0.1);"
        f"border-radius:8px;border:1px solid #4CAF50;'>"
        f"<div style='color:#4CAF50;font-weight:700;font-size:15px;'>✅ Generation complete</div>"
        f"<div style='color:#ccc;font-size:12px;margin-top:4px;'>"
        f"{done} records &nbsp;·&nbsp; {failed} failed &nbsp;·&nbsp; "
        f"<a href='https://huggingface.co/datasets/{DATASET_REPO}' target='_blank' "
        f"style='color:#63B3ED;'>View dataset →</a></div></div>"
    )
    yield final_html, "\n".join(log_lines)


# ─── Main evaluation function ────────────────────────────────────────────────


def render_reliability(rel_report, k: int) -> str:
    """Render pass@k / pass^k as an HTML table."""
    if not rel_report or not rel_report.evaluator_results:
        return ""
    rows = rel_report.summary_table()
    verdict_style = {
        "reliable": ("#4CAF50", "✅"),
        "unstable": ("#FF9800", "⚠️"),
        "unreliable": ("#F44336", "❌"),
    }
    header = (
        f"<h3 style='color:#63B3ED;margin:18px 0 10px;font-size:15px;'>"
        f"🔄 Reliability Testing — k={k} trials</h3>"
        f"<div style='background:rgba(99,179,237,0.08);border-radius:8px;padding:12px 16px;margin-bottom:10px;font-size:12px;color:#aaa;'>"
        f"<b>pass@{k}</b> = P(≥1 of {k} trials passes) — optimistic bound &nbsp;| "
        f"<b>pass^{k}</b> = P(ALL {k} trials pass) — reliability estimate</div>"
    )
    table = (
        "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
        "<thead><tr style='color:#aaa;border-bottom:1px solid rgba(255,255,255,0.1);'>"
        f"<th style='text-align:left;padding:6px 8px;'>Evaluator</th>"
        f"<th style='text-align:center;padding:6px 8px;'>Avg</th>"
        f"<th style='text-align:center;padding:6px 8px;'>pass@{k}</th>"
        f"<th style='text-align:center;padding:6px 8px;'>pass^{k}</th>"
        f"<th style='text-align:center;padding:6px 8px;'>Verdict</th>"
        "</tr></thead><tbody>"
    )
    for r in rows:
        color, icon = verdict_style.get(r["Verdict"], ("#888", "?"))
        table += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
            f"<td style='padding:5px 8px;color:#ddd;'>{r['Evaluator']}</td>"
            f"<td style='text-align:center;padding:5px 8px;color:#ccc;'>{r['Avg Score']}</td>"
            f"<td style='text-align:center;padding:5px 8px;color:#63B3ED;font-weight:600;'>{r[f'pass@{k}']}</td>"
            f"<td style='text-align:center;padding:5px 8px;color:{color};font-weight:700;'>{r[f'pass^{k}']}</td>"
            f"<td style='text-align:center;padding:5px 8px;'><span style='color:{color};'>{icon} {r['Verdict']}</span></td>"
            "</tr>"
        )
    table += "</tbody></table>"

    summary = (
        f"<div style='margin-top:10px;padding:10px 14px;background:rgba(255,255,255,0.05);"
        f"border-radius:6px;font-size:12px;color:#ccc;'>"
        f"Overall — pass@{k}: <b style='color:#63B3ED;'>{rel_report.overall_pass_at_k:.0%}</b>"
        f" &nbsp;| pass^{k}: <b style='color:#4CAF50;'>{rel_report.overall_pass_hat_k:.0%}</b>"
        f" &nbsp;| avg score: <b>{rel_report.avg_score:.0%}</b></div>"
    )
    return header + table + summary


def run_evaluation(
    trace_json: str,
    use_session: bool,
    use_trace: bool,
    use_span: bool,
    sel_session: list,
    sel_trace: list,
    sel_span: list,
    threshold: float,
    k_trials: int,
    eval_mode_radio: str,
    hf_token: str,
    exp_response: str,
    exp_trajectory: str,
    assertions_text: str,
    progress=gr.Progress(track_tqdm=True),
):
    # ── 1. Parse input ────────────────────────────────────────────────────
    progress(0.05, desc="Parsing trace…")
    try:
        session = parse_trace(trace_json)
    except Exception as e:
        err = (
            f"<div style='color:#F44336;padding:20px;'>❌ <b>Parse error:</b> {e}</div>"
        )
        return err, None, None, None, err

    # ── 2. Build ground truth ─────────────────────────────────────────────
    gt = None
    if exp_response.strip() or exp_trajectory.strip() or assertions_text.strip():
        traj = (
            [t.strip() for t in exp_trajectory.split(",") if t.strip()]
            if exp_trajectory.strip()
            else None
        )
        asrt = (
            [a.strip() for a in assertions_text.splitlines() if a.strip()]
            if assertions_text.strip()
            else None
        )
        gt = GroundTruth(
            expected_response=exp_response.strip() or None,
            expected_trajectory=traj,
            assertions=asrt,
        )

    # ── 3. Resolve selected evaluators ───────────────────────────────────
    sess_evals = sel_session if use_session else []
    trace_evals = sel_trace if use_trace else []
    span_evals = sel_span if use_span else []

    if not sess_evals and not trace_evals and not span_evals:
        warn = "<div style='color:#FF9800;padding:20px;'>⚠️ No evaluators selected — please enable at least one level.</div>"
        return warn, None, None, None, warn

    # ── 4. Build LLM judge (if requested) ────────────────────────────────
    use_llm = eval_mode_radio == "LLM Judge (QwQ-32B)"
    mode = EvalMode.LLM if use_llm else EvalMode.HEURISTIC
    judge = None
    if use_llm:
        token = hf_token.strip() or None
        judge = LLMJudge(api_key=token)
        if not judge.available:
            warn = "<div style='color:#FF9800;padding:20px;'>⚠️ LLM mode selected but no HF Token provided — falling back to heuritic.</div>"
            mode = EvalMode.HEURISTIC

    # ── 5. Run evaluation (single or k trials) ─────────────────────────────
    progress(0.15, desc="Running evaluators…")
    k = int(k_trials)
    runner = EvalRunner(
        selected_session_evals=sess_evals,
        selected_trace_evals=trace_evals,
        selected_span_evals=span_evals,
        threshold=threshold,
        mode=mode,
        llm_judge=judge,
    )

    if k > 1:
        progress(0.20, desc=f"Running {k} trials…")
        reports = runner.run_k_trials(session, gt, k=k)
        report = reports[0]  # use first for charts
        rel_report = compute_reliability(reports, threshold=threshold)
    else:
        report = runner.run(session, gt)
        rel_report = None

    progress(0.75, desc="Building visualizations…")

    # ── 5. Overall banner ─────────────────────────────────────────────────
    banner = render_overall_banner(report)

    # ── 6. Score cards HTML ───────────────────────────────────────────────
    cards_html = ""

    # SESSION
    if report.session_scores:
        cards_html += (
            "<h3 style='color:#9B59B6;margin:18px 0 10px;font-size:15px;'>"
            "📦 Session Level</h3>"
        )
        for sc in report.session_scores:
            cards_html += render_score_card(sc)

    # TRACE — grouped per turn
    if report.trace_scores:
        cards_html += (
            "<h3 style='color:#3498DB;margin:18px 0 10px;font-size:15px;'>"
            "🔄 Trace Level <span style='font-size:11px;color:rgba(150,150,200,0.7);'>"
            "(per conversation turn)</span></h3>"
        )
        by_trace: dict = {}
        for sc in report.trace_scores:
            by_trace.setdefault(sc.target_id, []).append(sc)

        for tid, t_scores in by_trace.items():
            avg_t = sum(x.score for x in t_scores) / len(t_scores)
            color_t = _bar_color(avg_t)
            cards_html += (
                f"<div style='color:rgba(180,180,200,0.7);font-size:11px;"
                f"margin:12px 0 5px;padding-bottom:4px;border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"Turn: <code>{tid}</code> &nbsp;·&nbsp; avg "
                f"<span style='color:{color_t};font-weight:700;'>{avg_t:.0%}</span></div>"
            )
            for sc in t_scores:
                cards_html += render_score_card(sc)

    # SPAN — grouped per span
    if report.span_scores:
        cards_html += (
            "<h3 style='color:#27AE60;margin:18px 0 10px;font-size:15px;'>"
            "🔧 Span Level <span style='font-size:11px;color:rgba(150,200,150,0.7);'>"
            "(per tool call)</span></h3>"
        )
        by_span: dict = {}
        for sc in report.span_scores:
            by_span.setdefault(sc.target_id, []).append(sc)

        for sid, s_scores in by_span.items():
            cards_html += (
                f"<div style='color:rgba(180,200,180,0.7);font-size:11px;"
                f"margin:12px 0 5px;padding-bottom:4px;border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"Span: <code>{sid}</code></div>"
            )
            for sc in s_scores:
                cards_html += render_score_card(sc)

    # ── 7. Charts ─────────────────────────────────────────────────────────
    avg_scores = report.avg_score_by_evaluator()
    radar = create_radar_chart(avg_scores)
    bar = create_bar_chart(report)
    heatmap = create_trace_timeline(report)

    rel_html = render_reliability(rel_report, k) if rel_report else ""
    progress(1.0, desc="Done!")
    return banner, radar, bar, heatmap, rel_html, cards_html


# ─── Gradio layout ───────────────────────────────────────────────────────────

_CSS = """
body, .gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
.gradio-container { max-width: 1060px !important; }
#run-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: white !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  border: none !important;
  padding: 12px 28px !important;
}
#run-btn:hover { opacity: 0.88; }
.gr-tab-nav { border-radius: 8px; }
footer { display: none !important; }
"""

_TITLE_HTML = """
<div style="text-align:center;padding:22px 0 14px;">
  <div style="font-size:30px;margin-bottom:6px;">🧪 AI Agent Evaluation Pipeline</div>
  <p style="color:rgba(180,180,180,0.85);margin:0;font-size:14px;max-width:600px;margin:0 auto;">
    Evaluate AI agents at <b style="color:#9B59B6;">Session</b>,
    <b style="color:#3498DB;">Trace</b>, and
    <b style="color:#27AE60;">Span</b> levels —
    inspired by <b>Amazon Bedrock AgentCore Evaluations</b>
  </p>
</div>
"""

_HOW_IT_WORKS = """
### How it works

| Level | Scope | Evaluators |
|-------|-------|------------|
| 📦 **Session** | Full conversation | Goal Success Rate |
| 🔄 **Trace** | Per turn (user → agent) | Helpfulness, Correctness, Coherence, Conciseness, Faithfulness, Harmfulness, Instruction Following, Response Relevance, Context Relevance, Refusal, Stereotyping |
| 🔧 **Span** | Per tool call | Tool Selection Accuracy, Tool Parameter Accuracy |

**Modes:** `heuristic` (offline, no API key) · `llm` (LLM-as-judge, coming soon)

**JSON format:** `session_id`, `user_goal`, `system_prompt`(opt), `traces[]` → `trace_id`, `user_input`, `agent_response`, `spans[]`
"""

# Build evaluator choice lists
_SESS_CHOICES = [(ALL_EVALUATORS[n].display_name, n) for n in SESSION_EVALUATORS]
_TRACE_CHOICES = [(ALL_EVALUATORS[n].display_name, n) for n in TRACE_EVALUATORS]
_SPAN_CHOICES = [(ALL_EVALUATORS[n].display_name, n) for n in SPAN_EVALUATORS]

with gr.Blocks(
    title="AI Agent Evaluation Pipeline",
) as demo:
    gr.HTML(_TITLE_HTML, padding=True)

    with gr.Tabs():
        # ── Tab 1: Load Trace ─────────────────────────────────────────────
        with gr.Tab("📥 Load Trace"):
            gr.Markdown("### Step 1 — Provide your agent trace")
            gr.Markdown(
                "Paste a JSON trace below, upload a file, or click a demo button to start immediately."
            )

            with gr.Row(equal_height=False):
                btn_simple = gr.Button("🎓 Simple Q&A", size="sm", variant="secondary")
                btn_tool = gr.Button("🔧 Tool Calling", size="sm", variant="secondary")
                btn_multi = gr.Button(
                    "🔄 Multi-turn + Tools", size="sm", variant="secondary"
                )

            trace_input = gr.Code(
                label="Agent Trace (JSON)",
                language="json",
                value=DEMO_MULTI_TURN,
                lines=22,
            )

            with gr.Accordion("🌲 Trace Preview", open=True):
                preview_md = gr.Markdown(parse_and_preview(DEMO_MULTI_TURN))

            # Wire demo buttons
            btn_simple.click(
                lambda: (DEMO_SIMPLE_QA, parse_and_preview(DEMO_SIMPLE_QA)),
                None,
                [trace_input, preview_md],
            )
            btn_tool.click(
                lambda: (DEMO_TOOL_CALLING, parse_and_preview(DEMO_TOOL_CALLING)),
                None,
                [trace_input, preview_md],
            )
            btn_multi.click(
                lambda: (DEMO_MULTI_TURN, parse_and_preview(DEMO_MULTI_TURN)),
                None,
                [trace_input, preview_md],
            )
            trace_input.change(parse_and_preview, trace_input, preview_md)

            with gr.Accordion("📖 JSON Schema Reference", open=False):
                gr.Code(
                    value=json.dumps(
                        {
                            "session_id": "my_session",
                            "user_goal": "Describe the overall goal of the user",
                            "system_prompt": "(optional) System instructions given to the agent",
                            "traces": [
                                {
                                    "trace_id": "t1",
                                    "user_input": "User's message",
                                    "agent_response": "Agent's reply",
                                    "retrieved_context": "(optional) RAG context",
                                    "spans": [
                                        {
                                            "span_id": "s1",
                                            "span_type": "TOOL_CALL",
                                            "tool_name": "my_tool",
                                            "tool_input": {"param": "value"},
                                            "tool_output": "Tool result string",
                                            "duration_ms": 250,
                                        }
                                    ],
                                }
                            ],
                        },
                        indent=2,
                    ),
                    language="json",
                    lines=10,
                )

        # ── Tab 2: Configure ──────────────────────────────────────────────
        with gr.Tab("⚙️ Configure"):
            gr.Markdown("### Step 2 — Choose evaluators and settings")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Evaluation Levels**")
                    use_session = gr.Checkbox(label="📦 Session Level", value=True)
                    use_trace = gr.Checkbox(label="🔄 Trace Level", value=True)
                    use_span = gr.Checkbox(
                        label="🔧 Span Level (tool calls)", value=True
                    )

                    gr.Markdown("**🤖 Evaluation Mode**")
                    eval_mode_radio = gr.Radio(
                        choices=["Heuristic (offline)", "LLM Judge (QwQ-32B)"],
                        value="Heuristic (offline)",
                        label="",
                        info="LLM mode requires a HuggingFace token with QwQ-32B access",
                    )
                    hf_token = gr.Textbox(
                        label="HF Token",
                        placeholder="hf_...",
                        type="password",
                        visible=False,
                    )
                    eval_mode_radio.change(
                        fn=lambda m: gr.update(visible=(m == "LLM Judge (QwQ-32B)")),
                        inputs=eval_mode_radio,
                        outputs=hf_token,
                    )

                    gr.Markdown("**Pass Threshold**")
                    threshold = gr.Slider(
                        minimum=0.30,
                        maximum=0.90,
                        step=0.05,
                        value=0.60,
                        label="Minimum score to pass",
                        info="Scores ≥ threshold are marked ✅ passed",
                    )

                    gr.Markdown("**🔄 Reliability Testing (pass@k / pass^k)**")
                    k_trials = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=1,
                        label="Trials (k)",
                        info="k=1 → standard mode. k>1 → runs multiple trials, shows pass@k & pass^k.",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("**📦 Session Evaluators** *(once per session)*")
                    sel_session = gr.CheckboxGroup(
                        choices=_SESS_CHOICES,
                        value=SESSION_EVALUATORS,
                        label="",
                    )

                    gr.Markdown(
                        "**🔄 Trace Evaluators** *(once per conversation turn)*"
                    )
                    sel_trace = gr.CheckboxGroup(
                        choices=_TRACE_CHOICES,
                        value=DEFAULT_TRACE_EVALS,
                        label="",
                    )

                    gr.Markdown("**🔧 Span Evaluators** *(once per tool call)*")
                    sel_span = gr.CheckboxGroup(
                        choices=_SPAN_CHOICES,
                        value=SPAN_EVALUATORS,
                        label="",
                    )

            with gr.Accordion(
                "📋 Ground Truth (Optional — improves scoring precision)", open=False
            ):
                gr.Markdown(
                    "Providing reference inputs enables ground-truth-based evaluation "
                    "(mirrors AgentCore's `expected_response`, `expected_trajectory`, and `assertions`)."
                )
                with gr.Row():
                    with gr.Column():
                        exp_response = gr.Textbox(
                            label="Expected Response",
                            placeholder="What should the final agent response look like?",
                            lines=3,
                        )
                        exp_trajectory = gr.Textbox(
                            label="Expected Tool Trajectory (comma-separated tool names)",
                            placeholder="search_restaurants, create_reservation",
                        )
                    with gr.Column():
                        assertions_text = gr.Textbox(
                            label="Assertions (one per line)",
                            placeholder="A restaurant reservation was made\nConfirmation number was provided\nThe restaurant matches user preferences",
                            lines=4,
                        )

            gr.Markdown("")
            run_btn = gr.Button(
                "▶ Run Evaluation", variant="primary", elem_id="run-btn", size="lg"
            )

        # ── Tab 3: Results ────────────────────────────────────────────────
        with gr.Tab("📊 Results"):
            overall_banner = gr.HTML(
                "<div style='text-align:center;color:rgba(150,150,150,0.6);padding:40px;'>"
                "← Configure and run evaluation to see results here</div>",
                padding=True,
            )

            with gr.Row():
                radar_chart = gr.Plot(label="🕸️ Evaluator Scores (Radar)", scale=1)
                bar_chart = gr.Plot(label="📊 Score Breakdown by Evaluator", scale=1)

            heatmap_chart = gr.Plot(label="🗋️ Score Heatmap: Evaluators × Turns")

            reliability_html = gr.HTML("", padding=True)
            score_cards_html = gr.HTML("", padding=True)

        # ── Tab 4: Dataset Generator ──────────────────────────────────────
        with gr.Tab("🗂️ Dataset"):
            gr.Markdown("### Generate Golden Dataset")
            gr.Markdown(
                f"Generates **(input, expected_output)** pairs for tech interview evaluation. "
                f"Uses `{GENERATOR_MODEL}`. "
                f"Results auto-uploaded to `{DATASET_REPO}`."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    domain_select = gr.CheckboxGroup(
                        choices=[(v, k) for k, v in DOMAIN_LABELS.items()],
                        value=list(DOMAIN_LABELS.keys()),
                        label="Domains",
                    )
                    records_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Records per domain",
                    )
                    hf_token_gen = gr.Textbox(
                        label="HF Token (requires Nemotron access)",
                        type="password",
                        placeholder="hf_...",
                    )
                    gen_btn = gr.Button(
                        "🚀 Generate & Upload Dataset", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    gen_status = gr.HTML(
                        "<div style='color:#888;padding:20px;text-align:center;'>"
                        "Configure and click Generate to start.</div>",
                        padding=True,
                    )
                    gen_log = gr.Textbox(
                        label="Progress",
                        lines=14,
                        interactive=False,
                        buttons=["copy"],
                    )

    # ── Tab 5: About ──────────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("ℹ️ About"):
            gr.Markdown(_HOW_IT_WORKS)
            gr.Markdown("""
### Evaluator Reference

| Evaluator | Level | Description |
|-----------|-------|-------------|
| **Goal Success Rate** | SESSION | Did the agent fully achieve the user's stated goal? |
| **Helpfulness** | TRACE | Does the response help the user progress toward their goal? |
| **Correctness** | TRACE | Is the response factually correct? (uses ground truth if provided) |
| **Coherence** | TRACE | Is the reasoning logically consistent and well-structured? |
| **Conciseness** | TRACE | Is the response appropriately brief without verbosity? |
| **Faithfulness** | TRACE | Is the response consistent with conversation history / context? |
| **Harmfulness** | TRACE | Does the response contain harmful or dangerous content? |
| **Instruction Following** | TRACE | Does the agent follow its system prompt instructions? |
| **Response Relevance** | TRACE | Does the response directly address what was asked? |
| **Context Relevance** | TRACE | Was the retrieved context relevant to the query? (RAG) |
| **Refusal Appropriateness** | TRACE | Did the agent correctly handle what to refuse? |
| **Stereotyping / Bias** | TRACE | Is there stereotypical or demographic bias? |
| **Tool Selection Accuracy** | SPAN | Did the agent choose the right tool? |
| **Tool Parameter Accuracy** | SPAN | Did the agent pass correct parameters to the tool? |

### Roadmap
- [ ] LLM-as-Judge mode (HuggingFace Inference API)
- [ ] OpenAI-compatible API support
- [ ] pass@k / pass^k reliability metrics
- [ ] Export results as JSON / CSV
- [ ] Custom evaluator builder (prompt templates)
- [ ] Dataset management for regression testing
            """)

    # ── Wire: Dataset generator ─────────────────────────────────────────────────
    gen_btn.click(
        fn=run_dataset_generation,
        inputs=[domain_select, records_slider, hf_token_gen],
        outputs=[gen_status, gen_log],
    )

    # ── Wire: Eval runner ──────────────────────────────────────────────────────
    run_btn.click(
        fn=run_evaluation,
        inputs=[
            trace_input,
            use_session,
            use_trace,
            use_span,
            sel_session,
            sel_trace,
            sel_span,
            threshold,
            k_trials,
            eval_mode_radio,
            hf_token,
            exp_response,
            exp_trajectory,
            assertions_text,
        ],
        outputs=[
            overall_banner,
            radar_chart,
            bar_chart,
            heatmap_chart,
            reliability_html,
            score_cards_html,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
        css=_CSS,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
