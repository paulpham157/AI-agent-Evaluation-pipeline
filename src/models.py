"""
Data models for the AI Agent Evaluation Pipeline.

Hierarchy mirrors Amazon Bedrock AgentCore Evaluations:
  Session → Trace → Span
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SpanType(str, Enum):
    TOOL_CALL = "TOOL_CALL"
    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"
    GUARDRAIL = "GUARDRAIL"
    OTHER = "OTHER"


class EvalLevel(str, Enum):
    SESSION = "SESSION"
    TRACE = "TRACE"
    SPAN = "SPAN"


class EvalMode(str, Enum):
    HEURISTIC = "heuristic"
    LLM = "llm"


# ─── Input Data Models ───


@dataclass
class Span:
    """A single atomic action within a trace (e.g. one tool call)."""

    span_id: str
    span_type: SpanType = SpanType.OTHER
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class Trace:
    """One round-trip: user sends a message → agent responds (may call tools)."""

    trace_id: str
    user_input: str
    agent_response: str
    spans: List[Span] = field(default_factory=list)
    system_prompt: Optional[str] = None  # Per-trace override
    retrieved_context: Optional[str] = None  # RAG context retrieved for this turn

    @property
    def tool_calls(self) -> List[Span]:
        return [s for s in self.spans if s.span_type == SpanType.TOOL_CALL]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Session:
    """The entire conversation: one or many traces between a user and an agent."""

    session_id: str
    user_goal: str
    traces: List[Trace] = field(default_factory=list)
    system_prompt: Optional[str] = None  # Default system prompt for all traces

    @property
    def conversation_history(self) -> str:
        turns = []
        for trace in self.traces:
            turns.append(f"User: {trace.user_input}")
            turns.append(f"Agent: {trace.agent_response}")
        return "\n\n".join(turns)

    @property
    def all_spans(self) -> List[Span]:
        return [span for trace in self.traces for span in trace.spans]

    @property
    def all_tool_calls(self) -> List[Span]:
        return [s for s in self.all_spans if s.span_type == SpanType.TOOL_CALL]


@dataclass
class GroundTruth:
    """Optional reference inputs for ground-truth-based evaluation."""

    expected_response: Optional[str] = None
    expected_trajectory: Optional[List[str]] = None  # Expected tool names in order
    assertions: Optional[List[str]] = None  # Natural language assertions


# ─── Output Data Models ───


@dataclass
class EvalScore:
    """Score produced by a single evaluator for a single target."""

    evaluator_name: str  # e.g. "helpfulness"
    evaluator_display: str  # e.g. "Helpfulness"
    level: EvalLevel
    score: float  # Normalized 0.0–1.0
    raw_score: float
    max_raw_score: float
    explanation: str
    passed: bool  # score >= threshold
    target_id: str  # session_id / trace_id / span_id
    target_label: str  # Human-readable target label
    mode: EvalMode = EvalMode.HEURISTIC

    @property
    def score_pct(self) -> int:
        return round(self.score * 100)

    @property
    def status_icon(self) -> str:
        if self.score >= 0.8:
            return "✅"
        elif self.score >= 0.6:
            return "⚠️"
        return "❌"

    @property
    def bar_color(self) -> str:
        if self.score >= 0.8:
            return "#4CAF50"
        elif self.score >= 0.6:
            return "#FF9800"
        return "#F44336"


@dataclass
class EvalReport:
    """Full evaluation results for one session."""

    session: Session
    scores: List[EvalScore] = field(default_factory=list)
    ground_truth: Optional[GroundTruth] = None
    eval_mode: EvalMode = EvalMode.HEURISTIC
    elapsed_seconds: float = 0.0

    @property
    def session_scores(self) -> List[EvalScore]:
        return [s for s in self.scores if s.level == EvalLevel.SESSION]

    @property
    def trace_scores(self) -> List[EvalScore]:
        return [s for s in self.scores if s.level == EvalLevel.TRACE]

    @property
    def span_scores(self) -> List[EvalScore]:
        return [s for s in self.scores if s.level == EvalLevel.SPAN]

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.passed) / len(self.scores)

    def scores_for_target(self, target_id: str) -> List[EvalScore]:
        return [s for s in self.scores if s.target_id == target_id]

    def avg_score_by_evaluator(self) -> Dict[str, float]:
        """Average score per evaluator across all targets."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for s in self.scores:
            totals[s.evaluator_display] = totals.get(s.evaluator_display, 0) + s.score
            counts[s.evaluator_display] = counts.get(s.evaluator_display, 0) + 1
        return {k: totals[k] / counts[k] for k in totals}
