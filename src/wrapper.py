"""
Agent Trace Wrapper
===================
Thin wrapper that captures AI agent conversations and auto-generates
evaluation-ready trace JSON — zero changes required to your agent code.

Usage (context manager):
    from src.wrapper import SessionTracer

    with SessionTracer(goal="Interview a Python candidate") as tracer:
        for user_msg in conversation:
            span = tracer.new_span()
            span.log_span("search_kb", {"query": user_msg}, kb_result)

            response = my_agent.invoke(user_msg)       # your agent, unchanged
            tracer.log_trace(user_msg, response, span)

        report = tracer.evaluate()
        print(f"Overall: {report.overall_score:.0%}")

Usage (decorator):
    @trace_agent(goal="Help user book a flight", auto_evaluate=True)
    def run_session(tracer):
        response = agent.invoke(user_input)
        tracer.log_trace(user_input, response)
        return response
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .models import EvalMode, GroundTruth


class SpanBuilder:
    """
    Collects tool-call spans for one conversation turn.
    Returned by SessionTracer.new_span() — call log_span() for each tool call.
    """

    def __init__(self):
        self._spans: List[Dict] = []

    def log_span(
        self,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Any = None,
        span_type: str = "TOOL_CALL",
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> "SpanBuilder":
        """Record one tool call (or retrieval, generation, etc.)."""
        span_id = f"s{len(self._spans) + 1}_{uuid.uuid4().hex[:5]}"
        self._spans.append(
            {
                "span_id": span_id,
                "span_type": span_type,
                "tool_name": tool_name,
                "tool_input": tool_input or {},
                "tool_output": str(tool_output) if tool_output is not None else None,
                "duration_ms": duration_ms,
                "error": error,
            }
        )
        return self

    def to_list(self) -> List[Dict]:
        return list(self._spans)


class SessionTracer:
    """
    Captures a full agent session as evaluation-ready JSON.

    Parameters
    ----------
    goal:          The user's overall goal for this session.
    session_id:    Auto-generated if omitted.
    system_prompt: System prompt given to the agent.
    """

    def __init__(
        self,
        goal: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.goal = goal
        self.session_id = (
            session_id
            or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.system_prompt = system_prompt
        self._traces: List[Dict] = []
        self._start_time: float = time.time()

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "SessionTracer":
        self._start_time = time.time()
        return self

    def __exit__(self, *_) -> bool:
        return False  # never suppress exceptions

    # ── Recording ────────────────────────────────────────────────────────

    def new_span(self) -> SpanBuilder:
        """Create a SpanBuilder for the current turn's tool calls."""
        return SpanBuilder()

    def log_trace(
        self,
        user_input: str,
        agent_response: str,
        span_builder: Optional[SpanBuilder] = None,
        trace_id: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ) -> "SessionTracer":
        """
        Record one conversation turn.

        Parameters
        ----------
        user_input:        What the user said.
        agent_response:    What the agent replied.
        span_builder:      SpanBuilder with tool calls made during this turn.
        trace_id:          Auto-assigned if omitted.
        retrieved_context: RAG context retrieved for this turn (optional).
        """
        idx = len(self._traces) + 1
        self._traces.append(
            {
                "trace_id": trace_id or f"t{idx}",
                "user_input": user_input,
                "agent_response": agent_response,
                "retrieved_context": retrieved_context,
                "spans": span_builder.to_list() if span_builder else [],
            }
        )
        return self

    # ── Export ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Return the session as a dict matching the pipeline's JSON schema."""
        d: Dict[str, Any] = {
            "session_id": self.session_id,
            "user_goal": self.goal,
            "traces": self._traces,
        }
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize the session to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> str:
        """
        Save the trace JSON to *path* and return the path.

        Example:
            tracer.save("traces/session_001.json")
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        return path

    # ── Evaluate ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        selected_session_evals: Optional[List[str]] = None,
        selected_trace_evals: Optional[List[str]] = None,
        selected_span_evals: Optional[List[str]] = None,
        ground_truth: Optional[GroundTruth] = None,
        threshold: float = 0.6,
    ):
        """
        Evaluate the captured session inline and return an EvalReport.

        Imports are lazy so this module stays importable without all deps.
        """
        if not self._traces:
            raise ValueError("No traces recorded yet. Call log_trace() at least once.")

        # Lazy imports to avoid circular deps at module load time
        from .parser import parse_trace
        from .runner import EvalRunner

        session = parse_trace(self.to_dict())
        runner = EvalRunner(
            selected_session_evals=selected_session_evals,
            selected_trace_evals=selected_trace_evals,
            selected_span_evals=selected_span_evals,
            threshold=threshold,
            mode=EvalMode.HEURISTIC,
        )
        return runner.run(session, ground_truth)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        """Number of turns recorded so far."""
        return len(self._traces)

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the tracer was created / entered."""
        return time.time() - self._start_time


# ── Decorator helper ─────────────────────────────────────────────────────────


def trace_agent(
    goal: str,
    system_prompt: Optional[str] = None,
    auto_evaluate: bool = False,
    save_path: Optional[str] = None,
):
    """
    Decorator factory that wraps an agent function with a SessionTracer.

    The wrapped function must accept a `tracer` as its *first* argument.

    Parameters
    ----------
    goal:          User's overall goal (required).
    system_prompt: System prompt for the agent.
    auto_evaluate: If True, evaluate after the function returns and print summary.
    save_path:     If set, save the trace JSON to this path automatically.

    Example:
        @trace_agent(goal="Answer user questions", auto_evaluate=True)
        def run_agent(tracer, user_input: str) -> str:
            response = my_llm.invoke(user_input)
            tracer.log_trace(user_input, response)
            return response

        answer = run_agent("What is photosynthesis?")
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tracer = SessionTracer(goal=goal, system_prompt=system_prompt)
            with tracer:
                result = func(tracer, *args, **kwargs)

            if save_path:
                tracer.save(save_path)

            if auto_evaluate:
                report = tracer.evaluate()
                print(
                    f"[eval] session={tracer.session_id}  "
                    f"overall={report.overall_score:.0%}  "
                    f"turns={tracer.turn_count}  "
                    f"time={tracer.elapsed:.2f}s"
                )
                return result, report

            return result

        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper

    return decorator
