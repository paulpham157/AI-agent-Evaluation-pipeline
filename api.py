"""
AI Agent Evaluation Pipeline — REST API
========================================
FastAPI server exposing the evaluation pipeline as HTTP endpoints.

Run:
    uvicorn api:app --reload --port 8000

Or:
    python api.py

Endpoints:
    GET  /health                   — liveness check
    GET  /evaluators               — list all 14 evaluators
    POST /evaluate                 — evaluate a full agent session
    POST /evaluate/quick           — evaluate with default evaluators (minimal payload)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluators import (
    ALL_EVALUATORS,
    DEFAULT_TRACE_EVALS,
    SESSION_EVALUATORS,
    SPAN_EVALUATORS,
    TRACE_EVALUATORS,
)
from src.llm_judge import LLMJudge
from src.models import EvalMode, GroundTruth
from src.parser import parse_trace
from src.runner import EvalRunner

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Agent Evaluation API",
    description=(
        "Evaluate AI agents at Session, Trace, and Span levels — "
        "inspired by Amazon Bedrock AgentCore Evaluations."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class GroundTruthInput(BaseModel):
    expected_response: Optional[str] = Field(
        None, description="Expected final agent response (enables Correctness scoring)"
    )
    expected_trajectory: Optional[List[str]] = Field(
        None, description="Expected tool call sequence, e.g. ['search', 'book']"
    )
    assertions: Optional[List[str]] = Field(
        None,
        description="Natural language assertions about the session outcome",
        examples=[["User's flight was booked", "Confirmation number was provided"]],
    )


class EvaluateRequest(BaseModel):
    hf_token: Optional[str] = Field(
        None, description="HuggingFace token for LLM judge mode"
    )
    mode: str = Field("heuristic", description="Evaluation mode: 'heuristic' or 'llm'")
    trace: Dict[str, Any] = Field(
        ...,
        description="Agent session trace in pipeline JSON format",
        examples=[
            {
                "session_id": "demo_001",
                "user_goal": "Book a table at an Italian restaurant",
                "traces": [
                    {
                        "trace_id": "t1",
                        "user_input": "I want Italian food tonight for 2",
                        "agent_response": "I found Flour + Water, available at 7 PM. Shall I book?",
                        "spans": [
                            {
                                "span_id": "s1",
                                "span_type": "TOOL_CALL",
                                "tool_name": "search_restaurants",
                                "tool_input": {"cuisine": "Italian", "party_size": 2},
                                "tool_output": '[{"name": "Flour + Water", "rating": 4.7}]',
                            }
                        ],
                    }
                ],
            }
        ],
    )
    session_evaluators: Optional[List[str]] = Field(
        None,
        description="Session-level evaluator names. Defaults to all session evaluators.",
    )
    trace_evaluators: Optional[List[str]] = Field(
        None,
        description="Trace-level evaluator names. Defaults to DEFAULT_TRACE_EVALS.",
    )
    span_evaluators: Optional[List[str]] = Field(
        None,
        description="Span-level evaluator names. Defaults to all span evaluators.",
    )
    ground_truth: Optional[GroundTruthInput] = None
    threshold: float = Field(
        0.6, ge=0.0, le=1.0, description="Pass/fail threshold (0–1)"
    )


class QuickEvaluateRequest(BaseModel):
    """Minimal payload — only requires user_goal + at least one trace."""

    trace: Dict[str, Any] = Field(..., description="Agent session trace JSON")
    threshold: float = Field(0.6, ge=0.0, le=1.0)


class EvaluatorInfo(BaseModel):
    name: str
    display: str
    description: str
    level: str


class ScoreItem(BaseModel):
    evaluator_name: str
    evaluator_display: str
    level: str
    score: float
    score_pct: int
    passed: bool
    explanation: str
    target_id: str
    target_label: str


class EvaluateResponse(BaseModel):
    session_id: str
    overall_score: float
    overall_score_pct: int
    pass_rate: float
    passed: bool
    elapsed_seconds: float
    total_evaluators: int
    level_averages: Dict[str, Optional[float]]
    summary: Dict[str, float]
    scores: List[ScoreItem]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_response(report, session_id: str) -> EvaluateResponse:
    """Convert an EvalReport into the API response model."""

    def avg(scores):
        return round(sum(s.score for s in scores) / len(scores), 4) if scores else None

    scores = [
        ScoreItem(
            evaluator_name=s.evaluator_name,
            evaluator_display=s.evaluator_display,
            level=s.level.value,
            score=round(s.score, 4),
            score_pct=s.score_pct,
            passed=s.passed,
            explanation=s.explanation,
            target_id=s.target_id,
            target_label=s.target_label,
        )
        for s in report.scores
    ]

    return EvaluateResponse(
        session_id=session_id,
        overall_score=round(report.overall_score, 4),
        overall_score_pct=round(report.overall_score * 100),
        pass_rate=round(report.pass_rate, 4),
        passed=report.overall_score >= 0.6,
        elapsed_seconds=report.elapsed_seconds,
        total_evaluators=len(report.scores),
        level_averages={
            "session": avg(report.session_scores),
            "trace": avg(report.trace_scores),
            "span": avg(report.span_scores),
        },
        summary={k: round(v, 4) for k, v in report.avg_score_by_evaluator().items()},
        scores=scores,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/evaluators", response_model=Dict[str, List[EvaluatorInfo]], tags=["Meta"])
def list_evaluators():
    """
    List all available evaluators grouped by level.

    Use the `name` values in EvaluateRequest.session_evaluators /
    trace_evaluators / span_evaluators to select specific evaluators.
    """

    def info(names, level):
        return [
            EvaluatorInfo(
                name=n,
                display=ALL_EVALUATORS[n].display_name,
                description=ALL_EVALUATORS[n].description,
                level=level,
            )
            for n in names
        ]

    return {
        "session": info(SESSION_EVALUATORS, "SESSION"),
        "trace": info(TRACE_EVALUATORS, "TRACE"),
        "span": info(SPAN_EVALUATORS, "SPAN"),
    }


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate(req: EvaluateRequest):
    """
    Evaluate an agent session at Session, Trace, and Span levels.

    - Omit `session_evaluators` / `trace_evaluators` / `span_evaluators`
      to use defaults.
    - Add `ground_truth` for more precise Correctness and GoalSuccessRate scoring.
    """
    try:
        session = parse_trace(req.trace)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    gt = None
    if req.ground_truth:
        gt = GroundTruth(
            expected_response=req.ground_truth.expected_response,
            expected_trajectory=req.ground_truth.expected_trajectory,
            assertions=req.ground_truth.assertions,
        )

    mode = EvalMode.LLM if req.mode == "llm" else EvalMode.HEURISTIC
    judge = None
    if mode == EvalMode.LLM:
        token = req.hf_token or os.getenv("HF_TOKEN", "")
        judge = LLMJudge(api_key=token)
        if not judge.available:
            raise HTTPException(
                status_code=422,
                detail="LLM mode requires hf_token or HF_TOKEN env var.",
            )

    runner = EvalRunner(
        selected_session_evals=req.session_evaluators,
        selected_trace_evals=req.trace_evaluators,
        selected_span_evals=req.span_evaluators,
        threshold=req.threshold,
        mode=mode,
        llm_judge=judge,
    )
    report = runner.run(session, gt)
    return _build_response(report, session.session_id)


@app.post("/evaluate/quick", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate_quick(req: QuickEvaluateRequest):
    """
    Quick evaluation with default evaluators — minimal payload required.

    Runs: GoalSuccessRate + 6 core trace evaluators + both span evaluators.
    """
    try:
        session = parse_trace(req.trace)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    runner = EvalRunner(threshold=req.threshold, mode=EvalMode.HEURISTIC)
    report = runner.run(session)
    return _build_response(report, session.session_id)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEV", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
