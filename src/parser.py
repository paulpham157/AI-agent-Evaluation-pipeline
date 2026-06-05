"""
Parse JSON agent traces into Session / Trace / Span data models.

Supported JSON format:
{
  "session_id": "...",
  "user_goal": "...",
  "system_prompt": "...",   // optional
  "traces": [
    {
      "trace_id": "...",
      "user_input": "...",
      "agent_response": "...",
      "system_prompt": "...",       // optional per-trace override
      "retrieved_context": "...",   // optional RAG context
      "spans": [
        {
          "span_id": "...",
          "span_type": "TOOL_CALL",  // TOOL_CALL | RETRIEVAL | GENERATION | GUARDRAIL | OTHER
          "tool_name": "...",
          "tool_input": { ... },
          "tool_output": "...",
          "duration_ms": 245,
          "error": null
        }
      ]
    }
  ]
}
"""

import json
from typing import Any, Dict, Union

from .models import Session, Span, SpanType, Trace


def parse_trace(data: Union[str, Dict]) -> Session:
    """Parse a JSON string or dict into a Session object."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError("Trace data must be a JSON object (dict)")

    session_id = data.get("session_id") or "session_001"
    user_goal = (data.get("user_goal") or "").strip()
    if not user_goal:
        raise ValueError("'user_goal' is required and cannot be empty")

    system_prompt = data.get("system_prompt")
    traces_raw = data.get("traces", [])

    if not isinstance(traces_raw, list) or len(traces_raw) == 0:
        raise ValueError("'traces' must be a non-empty array")

    traces = [_parse_trace(t, i, system_prompt) for i, t in enumerate(traces_raw)]

    return Session(
        session_id=session_id,
        user_goal=user_goal,
        traces=traces,
        system_prompt=system_prompt,
    )


def _parse_trace(data: Dict, index: int, session_system_prompt: str = None) -> Trace:
    trace_id = data.get("trace_id") or f"trace_{index + 1}"
    user_input = (data.get("user_input") or "").strip()
    agent_response = (data.get("agent_response") or "").strip()

    if not user_input:
        raise ValueError(f"Trace '{trace_id}' is missing 'user_input'")
    if not agent_response:
        raise ValueError(f"Trace '{trace_id}' is missing 'agent_response'")

    spans_raw = data.get("spans", [])
    spans = [_parse_span(s, j) for j, s in enumerate(spans_raw)]

    # Per-trace system_prompt overrides session-level
    system_prompt = data.get("system_prompt", session_system_prompt)

    return Trace(
        trace_id=trace_id,
        user_input=user_input,
        agent_response=agent_response,
        spans=spans,
        system_prompt=system_prompt,
        retrieved_context=data.get("retrieved_context"),
    )


def _parse_span(data: Dict, index: int) -> Span:
    span_id = data.get("span_id") or f"span_{index + 1}"

    span_type_str = (data.get("span_type") or "OTHER").upper().strip()
    try:
        span_type = SpanType(span_type_str)
    except ValueError:
        span_type = SpanType.OTHER

    duration = data.get("duration_ms")
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None

    return Span(
        span_id=span_id,
        span_type=span_type,
        tool_name=data.get("tool_name"),
        tool_input=data.get("tool_input"),
        tool_output=data.get("tool_output"),
        duration_ms=duration,
        error=data.get("error"),
    )


def format_trace_tree(session: Session) -> str:
    """Return a markdown-formatted tree overview of the session."""
    total_spans = len(session.all_spans)
    tool_calls = len(session.all_tool_calls)

    lines = [
        f"**📦 Session** `{session.session_id}`",
        f"**🎯 Goal:** {session.user_goal}",
        f"**📊 Stats:** {len(session.traces)} turn(s) · {total_spans} span(s) · {tool_calls} tool call(s)",
    ]

    if session.system_prompt:
        snippet = session.system_prompt[:100].replace("\n", " ")
        lines.append(
            f"**🤖 System Prompt:** *{snippet}{'...' if len(session.system_prompt) > 100 else ''}*"
        )

    lines.append("")

    for i, trace in enumerate(session.traces):
        icon = "🔧" if trace.has_tool_calls else "💬"
        in_snippet = trace.user_input[:70]
        out_snippet = trace.agent_response[:70]
        if len(trace.user_input) > 70:
            in_snippet += "…"
        if len(trace.agent_response) > 70:
            out_snippet += "…"

        lines.append(f"{icon} **Turn {i + 1}** (`{trace.trace_id}`)")
        lines.append(f"   👤 *{in_snippet}*")
        lines.append(f"   🤖 *{out_snippet}*")

        for span in trace.spans:
            span_icon = {
                "TOOL_CALL": "🔧",
                "RETRIEVAL": "🔍",
                "GENERATION": "✍️",
                "GUARDRAIL": "🛡️",
            }.get(span.span_type.value, "⚙️")
            label = span.tool_name or span.span_type.value
            dur = f" ({span.duration_ms:.0f}ms)" if span.duration_ms else ""
            err = " ⚠️ ERROR" if span.error else ""
            lines.append(f"   └─ {span_icon} `{span.span_id}`: **{label}**{dur}{err}")

        lines.append("")

    return "\n".join(lines)
