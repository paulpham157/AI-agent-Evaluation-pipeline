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

import gc
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure src/ is importable whether run from repo root or HF Spaces
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

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


from src.dataset_generator import (
    DOMAIN_LABELS,
    GENERATOR_MODEL,
    SCENARIOS,
    build_templates,
    generate_dataset,
)
from src.evaluators import (
    ALL_EVALUATORS,
    DEFAULT_TRACE_EVALS,
    SESSION_EVALUATORS,
    SPAN_EVALUATORS,
    TRACE_EVALUATORS,
)
from src.llm_judge import LLMJudge, LocalQwenJudge
from src.models import EvalLevel, EvalMode, GroundTruth
from src.parser import format_trace_tree, parse_trace
from src.reliability import compute_reliability
from src.runner import EvalRunner
from src.visualizer import create_bar_chart, create_radar_chart, create_trace_timeline

# ─── App state: loaded models (lazy, only after user saves) ─────────────────

_app_state = {
    "judge_mode": "local",
    "judge": None,
    "gen_backend": "llama-cpp",
    "gen_client": None,
    "gen_hf_token": None,
}


def unload_models():
    """Free GPU memory by deleting cached model instances."""
    for key in ("judge", "gen_client"):
        obj = _app_state.get(key)
        if obj is not None:
            try:
                if hasattr(obj, "_llm") and obj._llm is not None:
                    del obj._llm
            except Exception:
                pass
    _app_state["judge"] = None
    _app_state["gen_client"] = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def save_config(
    judge_mode: str,
    hf_token: str,
    gen_backend: str,
    gen_hf_token: str,
) -> str:
    """Validate and load models per user selection. Returns status HTML."""
    unload_models()
    status = []
    _app_state["judge_mode"] = "heuristic"
    _app_state["gen_backend"] = "llama-cpp"
    _app_state["gen_hf_token"] = gen_hf_token.strip() or None

    # ── Load judge ─────────────────────────────────────────────────────
    if judge_mode == "LLM Judge (Inference API)":
        token = hf_token.strip() or None
        judge = LLMJudge(api_key=token)
        if judge.available:
            _app_state["judge"] = judge
            _app_state["judge_mode"] = "inference"
            status.append("✅ Judge: Inference API (no local model needed)")
        else:
            status.append("⚠️ Judge: Inference API — no HF token, will use heuristic")
    elif judge_mode == "LLM Judge (Local Qwen3 8B)":
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            status.append("⏳ Installing llama-cpp-python (compiling may take 2–3 min)…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        judge = LocalQwenJudge(model_path=None)
        judge._init_llm()
        if judge.available:
            _app_state["judge"] = judge
            _app_state["judge_mode"] = "local"
            status.append("✅ Judge: Local Qwen3 8B loaded")
        else:
            status.append("❌ Judge: Local Qwen3 8B failed to load — will use heuristic")
    else:
        status.append("✅ Judge: Heuristic (no model needed)")

    # ── Load dataset generator ─────────────────────────────────────────
    if gen_backend == "llama-cpp":
        from src.dataset_generator import LlamaCppClient

        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            status.append("⏳ Installing llama-cpp-python (compiling may take 2–3 min)…")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        client = LlamaCppClient()
        try:
            client._init_llm()
            _app_state["gen_client"] = client
            _app_state["gen_backend"] = "llama-cpp"
            status.append("✅ Generator: llama.cpp model loaded")
        except RuntimeError as e:
            status.append(f"❌ Generator: llama.cpp not available — {e}")
    else:
        status.append("✅ Generator: Inference API (no local model needed)")

    return _config_status_html(status)


def _config_status_html(items: list[str]) -> str:
    rows = "".join(
        f"<div style='padding:4px 8px;font-size:13px;color:#ccc;'>{s}</div>"
        for s in items
    )
    return f"<div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:8px;'>{rows}</div>"


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


# ─── Dataset generation functions ──────────────────────────────────────────────


def run_generate(
    domains: list,
    records_per_domain: int,
    upload: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate golden dataset records with UI progress.

    Uses the backend and model pre-configured via ⚙️ Configure → Save.
    """
    if not domains:
        yield "<div style='color:#F44336;'>❌ Select at least one domain.</div>", ""
        return

    backend = _app_state["gen_backend"]
    log_lines = []
    log_lines.append(
        f"🚀 Starting generation: {len(domains)} domains × {records_per_domain} records"
        f"  |  backend: {backend}"
    )

    def progress_callback(current: int, total: int, msg: str):
        progress((current + 1) / total if total else 0, desc=f"Generating {msg}...")

    kwargs = dict(
        domains=domains,
        records_per_domain=records_per_domain,
        upload=upload,
        hf_token=_app_state.get("gen_hf_token"),
        progress_callback=progress_callback,
    )

    if backend == "llama-cpp" and _app_state["gen_client"] is not None:
        kwargs["client"] = _app_state["gen_client"]
        kwargs["backend"] = "llama-cpp"
    elif backend == "inference":
        kwargs["backend"] = "inference"
    else:
        kwargs["backend"] = backend

    records, failed, log = generate_dataset(**kwargs)

    log_lines.extend(log)
    passed = len(records) - len(failed)
    color = "#4CAF50" if failed == 0 else "#FF9800"
    result_html = f"""
<div style="padding:16px;background:rgba(99,179,237,0.08);border-radius:8px;border:1px solid rgba(99,179,237,0.2);">
  <div style="color:#63B3ED;font-weight:700;font-size:16px;margin-bottom:8px;">📊 Generation Complete</div>
  <div style="color:#ccc;font-size:13px;">
    ✅ Generated: <b style="color:#4CAF50;">{passed}</b>
    &nbsp;·&nbsp; ✗ Failed: <b style="color:{color};">{len(failed)}</b>
    &nbsp;·&nbsp; Total: <b>{len(records)}</b>
  </div>
</div>"""

    yield result_html, "\n".join(log_lines)


# ─── Benchmark functions ──────────────────────────────────────────────────────


def load_records_from_url(url: str) -> list:
    """Load JSONL records from a HF dataset repo URL (data/golden_dataset.jsonl)."""
    from urllib.parse import urlparse

    from huggingface_hub import hf_hub_download

    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc or "/datasets/" not in parsed.path:
        raise ValueError(f"Not a HF dataset URL: {url}")
    repo_id = parsed.path.split("/datasets/")[1].strip("/").split("/")[0]
    path = hf_hub_download(
        repo_id=repo_id,
        filename="data/golden_dataset.jsonl",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_pasted_jsonl(text: str) -> list:
    """Parse pasted JSONL content into list of records."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def call_openai_compat(
    url: str, scenario: dict, api_key: str, model: str, timeout: int = 60
) -> str:
    """POST to an OpenAI-compatible /v1/chat/completions endpoint."""
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    body = {
        "messages": [
            {"role": "system", "content": scenario.get("system_prompt", "")},
            {"role": "user", "content": scenario["initial_message"]},
        ],
    }
    if model.strip():
        body["model"] = model.strip()
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def build_trace_json(rec: dict, agent_response: str) -> str:
    """Build a parseable trace JSON from a dataset record + agent response."""
    scenario = rec.get("scenario", {})
    return json.dumps(
        {
            "session_id": rec.get("id", "unknown"),
            "user_goal": scenario.get("user_goal", ""),
            "system_prompt": scenario.get("system_prompt"),
            "traces": [
                {
                    "trace_id": "t1",
                    "user_input": scenario.get("initial_message", ""),
                    "agent_response": agent_response,
                }
            ],
        },
        ensure_ascii=False,
    )


def run_benchmark(
    dataset_url: str,
    pasted_jsonl: str,
    agent_url: str,
    api_key: str,
    model_name: str,
    use_session: bool,
    use_trace: bool,
    use_span: bool,
    sel_session: list,
    sel_trace: list,
    sel_span: list,
    threshold: float,
    progress=gr.Progress(track_tqdm=True),
):
    """Run benchmark: load dataset, call agent for each record, eval, aggregate."""

    def render_status(phase: str, done: int, total: int, current_id: str = "") -> str:
        pct = int(done / total * 100) if total else 0
        current = f" &nbsp;·&nbsp; ⏳ {current_id}" if current_id else ""
        return (
            f"<div style='padding:12px;background:rgba(255,255,255,0.05);border-radius:8px;'>"
            f"<div style='font-size:12px;color:#aaa;margin-bottom:6px;'>"
            f"<b>{phase}</b> &nbsp;·&nbsp; {done}/{total} ({pct}%){current}</div>"
            f"<div style='background:rgba(255,255,255,0.1);border-radius:3px;height:6px;'>"
            f"<div style='background:#63B3ED;height:6px;border-radius:3px;width:{pct}%;'></div>"
            f"</div></div>"
        )

    def render_table(rows: list) -> str:
        if not rows:
            return ""
        body = ""
        for r in rows:
            color = "#4CAF50" if r["passed"] else "#F44336"
            icon = "✅" if r["passed"] else "⚠️"
            score = r["score"]
            score_str = f"{score:.0%}" if isinstance(score, float) else "—"
            err_cell = (
                f"<div style='color:#F44336;font-size:10px;'>{r['error']}</div>"
                if r.get("error")
                else ""
            )
            body += (
                "<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
                f"<td style='padding:6px 8px;color:#ddd;font-size:12px;'>{r['id']}</td>"
                f"<td style='padding:6px 8px;color:#aaa;font-size:11px;'>{r['domain']}</td>"
                f"<td style='padding:6px 8px;color:#aaa;font-size:11px;'>{r['difficulty']}</td>"
                f"<td style='padding:6px 8px;text-align:center;color:{color};font-weight:700;'>{score_str} {icon}</td>"
                f"<td style='padding:6px 8px;'>{err_cell}</td>"
                "</tr>"
            )
        return (
            "<table style='width:100%;border-collapse:collapse;margin-top:14px;'>"
            "<thead><tr style='color:#aaa;border-bottom:1px solid rgba(255,255,255,0.1);font-size:11px;'>"
            "<th style='text-align:left;padding:6px 8px;'>ID</th>"
            "<th style='text-align:left;padding:6px 8px;'>Domain</th>"
            "<th style='text-align:left;padding:6px 8px;'>Difficulty</th>"
            "<th style='text-align:center;padding:6px 8px;'>Score</th>"
            "<th style='text-align:left;padding:6px 8px;'>Error</th>"
            "</tr></thead><tbody>" + body + "</tbody></table>"
        )

    def render_aggregate(rows: list, total: int) -> str:
        scored = [r for r in rows if isinstance(r["score"], float)]
        if not scored:
            return ""
        ok = sum(1 for r in scored if r["passed"])
        avg = sum(r["score"] for r in scored) / len(scored)
        by_domain: dict = {}
        for r in scored:
            d = r["domain"] or "—"
            by_domain.setdefault(d, []).append(r["score"])
        domain_chips = " ".join(
            f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:3px 9px;"
            f"background:rgba(255,255,255,0.07);border-radius:10px;font-size:11px;color:#ccc;'>"
            f"{d}: <b style='color:#4CAF50;'>{sum(s)/len(s):.0%}</b></span>"
            for d, s in sorted(by_domain.items())
        )
        return (
            f"<div style='margin-top:16px;padding:14px;background:rgba(99,179,237,0.08);"
            f"border-radius:8px;border:1px solid rgba(99,179,237,0.2);'>"
            f"<div style='color:#63B3ED;font-weight:700;font-size:14px;margin-bottom:8px;'>📊 Aggregate</div>"
            f"<div style='color:#ccc;font-size:12px;margin-bottom:6px;'>"
            f"Passed: <b style='color:#4CAF50;'>{ok}/{len(scored)}</b> "
            f"&nbsp;·&nbsp; Avg: <b style='color:#4CAF50;'>{avg:.0%}</b>"
            f"&nbsp;·&nbsp; Threshold: {threshold:.0%}</div>"
            f"<div style='color:#aaa;font-size:11px;'>{domain_chips}</div></div>"
        )

    def panel(*htmls: str) -> str:
        return "".join(h for h in htmls if h)

    progress(0.02, desc="Loading dataset…")
    yield panel(render_status("Loading dataset", 0, 1)), "📂 Loading dataset…"
    try:
        if pasted_jsonl.strip():
            records = parse_pasted_jsonl(pasted_jsonl)
            source = "pasted JSONL"
        else:
            records = load_records_from_url(dataset_url.strip())
            source = dataset_url.strip()
    except Exception as e:
        err = f"❌ Failed to load dataset: {e}"
        yield (
            panel(f"<div style='color:#F44336;padding:14px;'>{err}</div>"),
            f"ERROR: {e}\nPaste JSONL directly if the URL is empty or unreachable.",
        )
        return

    if not records:
        yield (
            panel("<div style='color:#FF9800;padding:14px;'>⚠️ Dataset loaded but empty.</div>"),
            "No records found in source.",
        )
        return

    total = len(records)
    log_lines = [f"✅ Loaded {total} records from {source}"]
    yield (
        panel(
            render_status("Loaded", total, total),
            f"<div style='color:#4CAF50;padding:10px;'>📂 {total} records loaded from {source}</div>",
        ),
        "\n".join(log_lines),
    )

    if not agent_url.strip():
        yield (
            panel("<div style='color:#F44336;padding:14px;'>❌ Agent URL is empty.</div>"),
            "ERROR: Provide an OpenAI-compatible chat completions URL.",
        )
        return

    sess_evals = sel_session if use_session else []
    trace_evals = sel_trace if use_trace else []
    span_evals = sel_span if use_span else []
    runner = EvalRunner(
        selected_session_evals=sess_evals,
        selected_trace_evals=trace_evals,
        selected_span_evals=span_evals,
        threshold=threshold,
        mode=EvalMode.HEURISTIC,
    )

    results = []
    for i, rec in enumerate(records):
        rid = rec.get("id", f"rec_{i}")
        domain = rec.get("domain", "")
        difficulty = rec.get("difficulty", "")
        progress(0.1 + 0.85 * i / total, desc=f"Running {rid}…")
        log_lines.append(f"⏳ {rid} ({domain}/{difficulty})…")
        yield (
            panel(render_status("Running", i, total, rid), render_table(results)),
            "\n".join(log_lines),
        )

        try:
            scenario = rec.get("scenario") or {}
            agent_out = call_openai_compat(
                agent_url.strip(),
                scenario,
                api_key or "",
                model_name or "",
                timeout=60,
            )
            trace_json = build_trace_json(rec, agent_out)
            session = parse_trace(trace_json)
            gt_data = rec.get("ground_truth") or {}
            gt = GroundTruth(
                expected_response=gt_data.get("expected_response"),
                expected_trajectory=gt_data.get("expected_trajectory"),
                assertions=gt_data.get("assertions"),
            )
            report = runner.run(session, gt)
            score = report.overall_score
            results.append(
                {
                    "id": rid,
                    "domain": domain,
                    "difficulty": difficulty,
                    "score": score,
                    "passed": score >= threshold,
                    "error": None,
                }
            )
            log_lines[-1] = f"✅ {rid} — {score:.0%}"
        except Exception as e:
            results.append(
                {
                    "id": rid,
                    "domain": domain,
                    "difficulty": difficulty,
                    "score": None,
                    "passed": False,
                    "error": f"{type(e).__name__}: {str(e)[:80]}",
                }
            )
            log_lines[-1] = f"✗ {rid} — {type(e).__name__}: {str(e)[:60]}"

        yield (
            panel(render_status("Running", i + 1, total), render_table(results)),
            "\n".join(log_lines),
        )

    progress(1.0, desc="Done!")
    yield (
        panel(
            render_status("Done", total, total),
            render_table(results),
            render_aggregate(results, total),
        ),
        "\n".join(log_lines),
    )


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

    # ── 4. Resolve LLM judge (pre-loaded from config, or lazy) ──────────
    use_llm = eval_mode_radio.startswith("LLM Judge")
    mode = EvalMode.LLM if use_llm else EvalMode.HEURISTIC
    judge = None
    if use_llm:
        # Use pre-loaded judge from config if available
        if _app_state["judge"] is not None and _app_state["judge_mode"] != "heuristic":
            judge = _app_state["judge"]
        elif eval_mode_radio == "LLM Judge (Inference API)":
            token = hf_token.strip() or None
            judge = LLMJudge(api_key=token)
            if not judge.available:
                warn = "<div style='color:#FF9800;padding:20px;'>⚠️ LLM Judge (Inference API) selected but no HF Token — falling back to heuristic.</div>"
                mode = EvalMode.HEURISTIC
        else:
            judge = LocalQwenJudge(model_path=None)
            if not judge.available:
                judge._init_llm()
            if not judge.available:
                warn = "<div style='color:#FF9800;padding:20px;'>⚠️ LLM Judge (Local) selected but model could not load — falling back to heuristic.</div>"
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
            gr.Markdown("### Step 1 — Model Configuration")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**🤖 LLM Judge Backend**")
                    cfg_judge_mode = gr.Radio(
                        choices=["Heuristic (offline)", "LLM Judge (Inference API)", "LLM Judge (Local Qwen3 8B)"],
                        value="LLM Judge (Local Qwen3 8B)",
                        label="",
                    )
                    cfg_judge_hf_token = gr.Textbox(
                        label="HF Token (for Inference API judge)",
                        placeholder="hf_...",
                        type="password",
                        visible=False,
                    )
                    cfg_judge_mode.change(
                        fn=lambda mode: gr.update(visible=(mode == "LLM Judge (Inference API)")),
                        inputs=cfg_judge_mode,
                        outputs=cfg_judge_hf_token,
                    )
                    gr.Markdown(
                        "> Local judge auto-downloads **Qwen3-8B Q4_K_M** (~5GB) from "
                        "[Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) on first use."
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**📦 Dataset Generator Backend**")
                    cfg_gen_backend = gr.Radio(
                        choices=["Inference API", "llama.cpp"],
                        value="llama.cpp",
                        label="",
                    )
                    cfg_gen_hf_token = gr.Textbox(
                        label="HF Token (for Inference API + upload)",
                        placeholder="hf_...",
                        type="password",
                        visible=False,
                    )
                    def _toggle_gen_fields(backend):
                        return gr.update(visible=(backend == "Inference API"))
                    cfg_gen_backend.change(
                        fn=_toggle_gen_fields,
                        inputs=cfg_gen_backend,
                        outputs=cfg_gen_hf_token,
                    )

            cfg_save_btn = gr.Button("💾 Save & Load Models", variant="primary", size="lg")
            cfg_status = gr.HTML(
                "<div style='color:#888;padding:10px;text-align:center;'>"
                "Configure models above and click Save to load.</div>",
                padding=True,
            )

            gr.Markdown("---")
            gr.Markdown("### Step 2 — Evaluator Selection & Settings")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Evaluation Levels**")
                    use_session = gr.Checkbox(label="📦 Session Level", value=True)
                    use_trace = gr.Checkbox(label="🔄 Trace Level", value=True)
                    use_span = gr.Checkbox(
                        label="🔧 Span Level (tool calls)", value=True
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

        # ── Tab 4: Benchmark ───────────────────────────────────────────────
        with gr.Tab("🧪 Benchmark"):
            gr.Markdown("### 🧪 Benchmark — evaluate your agent against a dataset")
            gr.Markdown(
                "Each record's `initial_message` is POSTed to your agent (OpenAI-compatible "
                "chat completions endpoint), the response is parsed into a trace, and all "
                "selected evaluators run. Ground truth from the record is used automatically."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**📦 Dataset**")
                    bm_dataset_url = gr.Textbox(
                        label="HF Dataset URL (loads data/golden_dataset.jsonl)",
                        value="https://huggingface.co/datasets/build-small-hackathon/agent-eval-golden-dataset",
                        placeholder="https://huggingface.co/datasets/...",
                    )
                    with gr.Accordion("📝 Or paste JSONL directly", open=False):
                        bm_paste_jsonl = gr.Textbox(
                            label="JSONL records",
                            lines=8,
                            placeholder='{"id":"python_001","scenario":{...},"ground_truth":{...}}\n...',
                        )
                    bm_load_btn = gr.Button(
                        "🔄 Load Dataset", variant="secondary", size="sm"
                    )
                    bm_records_info = gr.HTML(
                        "<div style='color:#888;padding:10px;'>No dataset loaded yet.</div>",
                        padding=True,
                    )

                    gr.Markdown("**🤖 Agent (OpenAI-compatible)**")
                    bm_agent_url = gr.Textbox(
                        label="Chat completions URL",
                        placeholder="https://your-agent.example.com/v1/chat/completions",
                    )
                    bm_api_key = gr.Textbox(
                        label="API Key (optional)",
                        type="password",
                        placeholder="Bearer xyz",
                    )
                    bm_model_name = gr.Textbox(
                        label="Model name (optional, sent in body if provided)",
                        placeholder="gpt-4o-mini",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**⚙️ Eval settings**")
                    bm_use_session = gr.Checkbox(label="📦 Session Level", value=True)
                    bm_use_trace = gr.Checkbox(label="🔄 Trace Level", value=True)
                    bm_use_span = gr.Checkbox(
                        label="🔧 Span Level (tool calls)", value=True
                    )
                    bm_sel_session = gr.CheckboxGroup(
                        choices=_SESS_CHOICES,
                        value=SESSION_EVALUATORS,
                        label="Session evaluators",
                    )
                    bm_sel_trace = gr.CheckboxGroup(
                        choices=_TRACE_CHOICES,
                        value=DEFAULT_TRACE_EVALS,
                        label="Trace evaluators",
                    )
                    bm_sel_span = gr.CheckboxGroup(
                        choices=_SPAN_CHOICES,
                        value=SPAN_EVALUATORS,
                        label="Span evaluators",
                    )
                    bm_threshold = gr.Slider(
                        minimum=0.30,
                        maximum=0.90,
                        step=0.05,
                        value=0.60,
                        label="Pass threshold",
                    )
                    bm_run_btn = gr.Button(
                        "🚀 Run Benchmark",
                        variant="primary",
                        size="lg",
                        elem_id="run-btn",
                    )

            with gr.Row():
                bm_results = gr.HTML(
                    "<div style='color:#888;padding:30px;text-align:center;'>"
                    "Load a dataset and click Run Benchmark to start.</div>",
                    padding=True,
                )

            with gr.Row():
                bm_log = gr.Textbox(
                    label="Log", lines=10, interactive=False, buttons=["copy"]
                )

        # ── Tab 5: Generate Dataset ────────────────────────────────────────────
        with gr.Tab("📦 Generate Dataset"):
            gr.Markdown("### 📦 Generate a golden benchmark dataset")
            gr.Markdown(
                "Use an LLM to generate golden (input, expected_output) records "
                "for evaluating AI tech interviewers across multiple domains."
            )
            gr.Markdown(
                "> **Backend & model** configured in ⚙️ Configure → Save & Load Models before generating."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**📐 Settings**")
                    gen_records = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Records per domain",
                    )
                    gen_upload = gr.Checkbox(
                        label="Upload to HF dataset repo after generation",
                        value=False,
                    )

                with gr.Column(scale=2):
                    gr.Markdown("**🌐 Domains**")
                    gen_domains = gr.CheckboxGroup(
                        choices=[(v, k) for k, v in DOMAIN_LABELS.items()],
                        value=list(SCENARIOS.keys()),
                        label="",
                    )

            gen_run_btn = gr.Button(
                "🚀 Generate Dataset", variant="primary", elem_id="run-btn", size="lg"
            )

            with gr.Row():
                gen_result = gr.HTML(
                    "<div style='color:#888;padding:20px;text-align:center;'>"
                    "Configure and click Generate to start.</div>",
                    padding=True,
                )

            with gr.Row():
                gen_log = gr.Textbox(
                    label="Log", lines=10, interactive=False, buttons=["copy"]
                )
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
- [x] LLM-as-Judge mode (HuggingFace Inference API)
- [ ] OpenAI-compatible API support
- [x] pass@k / pass^k reliability metrics
- [ ] Export results as JSON / CSV
- [ ] Custom evaluator builder (prompt templates)
- [x] Dataset management for regression testing (🧪 Benchmark tab)
            """)

    # ── Wire: Benchmark ────────────────────────────────────────────────────────
    def _preview_dataset(url, paste):
        try:
            if paste.strip():
                records = parse_pasted_jsonl(paste)
                src = "pasted JSONL"
            else:
                records = load_records_from_url(url.strip())
                src = url.strip()
            if not records:
                return "<div style='color:#FF9800;padding:10px;'>⚠️ Loaded 0 records.</div>"
            domains = sorted({r.get("domain", "") for r in records if r.get("domain")})
            return (
                f"<div style='color:#4CAF50;padding:10px;'>"
                f"📂 {len(records)} records loaded from {src}"
                f"<br><span style='color:#aaa;font-size:11px;'>"
                f"Domains: {', '.join(domains)}</span></div>"
            )
        except Exception as e:
            return f"<div style='color:#F44336;padding:10px;'>❌ {e}</div>"

    bm_load_btn.click(
        _preview_dataset,
        inputs=[bm_dataset_url, bm_paste_jsonl],
        outputs=bm_records_info,
    )

    bm_run_btn.click(
        fn=run_benchmark,
        inputs=[
            bm_dataset_url,
            bm_paste_jsonl,
            bm_agent_url,
            bm_api_key,
            bm_model_name,
            bm_use_session,
            bm_use_trace,
            bm_use_span,
            bm_sel_session,
            bm_sel_trace,
            bm_sel_span,
            bm_threshold,
        ],
        outputs=[bm_results, bm_log],
    )

    # ── Wire: Save Configuration ───────────────────────────────────────────────
    cfg_save_btn.click(
        fn=save_config,
        inputs=[
            cfg_judge_mode,
            cfg_judge_hf_token,
            cfg_gen_backend,
            cfg_gen_hf_token,
        ],
        outputs=cfg_status,
    )

    # ── Wire: Generate Dataset ──────────────────────────────────────────────────
    gen_run_btn.click(
        fn=run_generate,
        inputs=[
            gen_domains,
            gen_records,
            gen_upload,
        ],
        outputs=[gen_result, gen_log],
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
            cfg_judge_mode,
            cfg_judge_hf_token,
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
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=os.getenv("GRADIO_SHOW_ERROR", "true").lower() == "true",
    )
