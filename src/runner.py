"""
EvalRunner — orchestrates the evaluation of an entire session.

Runs the selected evaluators at each level (Session → Trace → Span)
and returns a consolidated EvalReport.
"""

import time
from typing import List, Optional

from .evaluators import (
    ALL_EVALUATORS,
    DEFAULT_TRACE_EVALS,
    SESSION_EVALUATORS,
    SPAN_EVALUATORS,
)
from .models import EvalLevel, EvalMode, EvalReport, EvalScore, GroundTruth, Session

try:
    from .llm_judge import LLMJudge as _LLMJudge
except ImportError:
    _LLMJudge = None  # type: ignore


class EvalRunner:
    """
    Orchestrates evaluation across all three levels.

    Usage:
        runner = EvalRunner(selected_trace_evals=["helpfulness", "coherence"])
        report = runner.run(session, ground_truth)
    """

    def __init__(
        self,
        selected_session_evals: Optional[List[str]] = None,
        selected_trace_evals: Optional[List[str]] = None,
        selected_span_evals: Optional[List[str]] = None,
        threshold: float = 0.6,
        mode: EvalMode = EvalMode.HEURISTIC,
        llm_judge=None,
    ):
        self.threshold = threshold
        self.mode = mode
        self.llm_judge = llm_judge  # LLMJudge instance or None

        # Fall back to defaults if None
        sess_names = (
            selected_session_evals
            if selected_session_evals is not None
            else SESSION_EVALUATORS
        )
        trace_names = (
            selected_trace_evals
            if selected_trace_evals is not None
            else DEFAULT_TRACE_EVALS
        )
        span_names = (
            selected_span_evals if selected_span_evals is not None else SPAN_EVALUATORS
        )

        # Instantiate evaluators, skip unknown names gracefully
        self._session_evals = self._instantiate(sess_names)
        self._trace_evals = self._instantiate(trace_names)
        self._span_evals = self._instantiate(span_names)

    def _instantiate(self, names: List[str]) -> dict:
        result = {}
        for name in names:
            if name in ALL_EVALUATORS:
                result[name] = ALL_EVALUATORS[name]()
        return result

    def run(
        self, session: Session, ground_truth: Optional[GroundTruth] = None
    ) -> EvalReport:
        """Run all selected evaluators and return a full EvalReport."""
        start = time.time()
        scores: List[EvalScore] = []

        judge = self.llm_judge if self.mode == EvalMode.LLM else None

        # ── SESSION level ──────────────────────────────────────────────────
        for name, ev in self._session_evals.items():
            try:
                score = ev.evaluate(
                    session,
                    ground_truth=ground_truth,
                    threshold=self.threshold,
                    mode=self.mode,
                    llm_judge=judge,
                )
                scores.append(score)
            except Exception:
                pass

        # ── TRACE level  ──────────────────────────────────────────────────
        for trace in session.traces:
            for name, ev in self._trace_evals.items():
                try:
                    score = ev.evaluate(
                        trace,
                        session,
                        ground_truth=ground_truth,
                        threshold=self.threshold,
                        mode=self.mode,
                        llm_judge=judge,
                    )
                    scores.append(score)
                except Exception:
                    pass

        # ── SPAN level  ───────────────────────────────────────────────────
        for trace in session.traces:
            for span in trace.spans:
                for name, ev in self._span_evals.items():
                    try:
                        score = ev.evaluate(
                            span,
                            trace,
                            session,
                            ground_truth=ground_truth,
                            threshold=self.threshold,
                            mode=self.mode,
                            llm_judge=judge,
                        )
                        scores.append(score)
                    except Exception:
                        pass

        return EvalReport(
            session=session,
            scores=scores,
            ground_truth=ground_truth,
            eval_mode=self.mode,
            elapsed_seconds=round(time.time() - start, 2),
        )

    def run_k_trials(
        self,
        session: Session,
        ground_truth: Optional[GroundTruth] = None,
        k: int = 3,
    ) -> list:
        """Run k independent evaluation trials and return a list of EvalReports."""
        return [self.run(session, ground_truth) for _ in range(k)]
