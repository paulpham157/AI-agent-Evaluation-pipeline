# AGENTS.md

Repository: **AI Agent Evaluation Pipeline** — evaluates AI agents at Session → Trace → Span levels (mirrors Amazon Bedrock AgentCore Evaluations). Stack: Python, Gradio, FastAPI, Pydantic, Plotly, huggingface_hub. No Cursor rules, no Copilot instructions, no existing `AGENTS.md` — this file is the source of truth for agentic coding agents.

## Project Layout

```
app.py                            # Gradio UI entry point  (port 7860)
api.py                            # FastAPI REST server    (port 8000)
scripts/
  generate_golden_dataset.py      # CLI: argparse-based dataset generator
src/
  __init__.py                     # empty package marker
  models.py                       # Session / Trace / Span / EvalScore dataclasses + enums
  parser.py                       # JSON trace → Session objects
  evaluators.py                   # All 14 evaluators (1 session + 11 trace + 2 span)
  runner.py                       # EvalRunner orchestrator
  llm_judge.py                    # LLMJudge (HF Inference API, optional dep)
  reliability.py                  # pass@k / pass^k metrics
  visualizer.py                   # Plotly charts
  wrapper.py                      # SessionTracer context manager + @trace_agent decorator
demos/                            # 3 demo trace JSON files
dataset/                          # golden_dataset.jsonl (generated)
requirements.txt                  # pip-only deps
```

## Build / Install / Run

```bash
pip install -r requirements.txt          # install deps (Gradio, FastAPI, Plotly, HF Hub)
python app.py                            # Gradio UI → http://localhost:7860
python api.py                            # REST API → http://localhost:8000  (or)
uvicorn api:app --reload --port 8000     # dev mode with autoreload
python scripts/generate_golden_dataset.py --dry-run                         # preview dataset generation
python scripts/generate_golden_dataset.py --domains python_backend --records-per-domain 3
python scripts/generate_golden_dataset.py --upload                          # generate + push to HF
python scripts/generate_golden_dataset.py --backend llama-cpp --dry-run     # llama.cpp backend (3rd backend)
python scripts/generate_golden_dataset.py --backend vllm --dry-run          # vLLM backend
```

## Tests / Lint / Typecheck

**There is no test suite, no linter config, no type checker, and no CI in this repo.** Do not invent a `pytest` command — there are no test files. `__pycache__/` directories exist, so the modules have been executed locally. If you add tests, follow the existing module layout (suggest `tests/test_<module>.py`) and add a `pyproject.toml` with `[tool.pytest.ini_options]`. If you add linting, use `ruff` (fast, single binary) configured in `pyproject.toml` — do not introduce `flake8`/`black`/`pylint` separately.

## Code Style

### Imports & module layout
- Inside `src/`, use **relative imports** (`from .models import Session`).
- Top-level scripts (`app.py`, `api.py`, `scripts/*.py`) prepend the repo root to `sys.path` so they can `from src.<module> import ...`:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent))   # or .parent.parent for scripts/
  ```
- `llm_judge` is an **optional** dependency — guard it with `try: from .llm_judge import LLMJudge as _LLMJudge\nexcept ImportError: _LLMJudge = None` (see `runner.py:19-22`). Use the same pattern for any new optional dep.
- Prefer **lazy imports inside methods** to break circular dependencies (see `wrapper.py:194-195`, `app.py`'s HF imports).
- `from __future__ import annotations` is **not** used — match the existing style (no `str | None`, use `Optional[str]`).

### Types
- Type hints everywhere on public functions, dataclass fields, and method signatures.
- `typing` module style: `Optional[X]`, `List[X]`, `Dict[K, V]`, `Tuple[A, B]`, `Any`. Do **not** switch to PEP 604 unions (`X | None`) — codebase is consistently pre-3.10.
- `dataclass` + `field(default_factory=list)` for mutable defaults; plain `=` for immutable defaults.
- Enums inherit from `(str, Enum)` so values serialize as strings: `class SpanType(str, Enum): TOOL_CALL = "TOOL_CALL"`.

### Naming
- Files & modules: `snake_case.py`.
- Classes: `PascalCase` (e.g., `EvalRunner`, `GoalSuccessRateEvaluator`, `BaseEvaluator`).
- Functions/methods/variables: `snake_case`.
- Module-level constants & registries: `UPPER_SNAKE_CASE` (`ALL_EVALUATORS`, `DEFAULT_TRACE_EVALS`, `SESSION_EVALUATORS`, `SPAN_EVALUATORS`, `_STOP_WORDS`, `_HARMFUL_PATTERNS`).
- Private helpers: `_` prefix (`_tokenize`, `_jaccard`, `_score_color`, `_empty_fig`).

### Formatting
- 4-space indentation, double quotes, ~100-120 char soft limit (no enforced line length — match neighboring code).
- Long f-strings and regex strings are fine on multiple lines; no trailing whitespace.
- Section dividers in long files use `# ─── Section Name ───` (Unicode box-drawing chars). See `evaluators.py` for examples.
- Emoji in **user-facing output** strings is welcome (✅ ⚠️ ❌ 📦 🔄 🔧) — the UI is intentionally visual.

### Comments & docstrings
- The user explicitly does **not** want comments added during normal work. Match the existing module docstrings (module-level + class-level) but don't add inline `# explanation` comments.
- Module docstrings: short prose with a section break and usage example (see `wrapper.py`, `llm_judge.py`, `reliability.py`).
- Class docstrings: 1-line summary, then `Parameters` / `Returns` / `Example` sections (NumPy-style in `llm_judge.py`, `wrapper.py`).

### Evaluator contract (most important pattern)
Every evaluator in `src/evaluators.py` follows this template. **Match it exactly** when adding a new one:
```python
class FooEvaluator(BaseEvaluator):
    name = "foo"                          # snake_case id, unique in ALL_EVALUATORS
    display_name = "Foo"                  # human label for UI
    level = EvalLevel.TRACE               # SESSION | TRACE | SPAN
    description = "..."                   # one-line, used by /evaluators endpoint
    llm_prompt_template = "..."           # str.format(**ctx) template for LLM mode

    def evaluate(self, target, ..., threshold=0.6, mode=EvalMode.HEURISTIC, llm_judge=None) -> EvalScore:
        if mode == EvalMode.LLM and llm_judge is not None:
            score, explanation = self._run_llm(llm_judge, **ctx)
        else:
            score, explanation = self._heuristic(...)
        return self._make_score(score, explanation, target_id, target_label, threshold, mode)

    def _heuristic(self, ...) -> Tuple[float, str]:
        ...
```
Then **register** in the `ALL_EVALUATORS` dict at the bottom of `evaluators.py` and add its name to the appropriate `SESSION_EVALUATORS` / `TRACE_EVALUATORS` / `SPAN_EVALUATORS` / `DEFAULT_TRACE_EVALS` list. Forgetting registration makes the evaluator unreachable from the UI / API.

### Scores, thresholds, normalization
- `_make_score` (in `BaseEvaluator`) normalizes via `max(0.0, min(1.0, raw / max_raw))`. Return a raw score in any range — `_make_score` handles clamping.
- Heuristics return `(score: float, explanation: str)` where `score` is already in `[0.0, 1.0]`.
- **Default pass threshold is `0.6`**; passing UI boundary is `0.8` (see `_score_color` in `models.py`, `visualizer.py`, `app.py`).
- LLM judge returns 1–5; `_clamp(score / 5.0)` converts to 0–1 (`llm_judge.py:19`).

### Error handling
- **Intentional graceful degradation**: in `runner.py`, each evaluator call is wrapped in `try/except Exception: pass` so one bad evaluator cannot fail the whole run. Match this pattern for new orchestrator code.
- API layer: re-raise parse errors as `HTTPException(status_code=422, detail=str(exc))` (see `api.py:259-260`).
- LLM judge: never raise — return `(0.5, f"LLM judge error ({type(exc).__name__}): {exc}")` so the heuristic path is the fallback.

### JSON & I/O
- Always pass `ensure_ascii=False` to `json.dumps` (the project's author and content are Vietnamese-friendly).
- Read text files with explicit `encoding="utf-8"`.

## Architecture Notes

- **3-level hierarchy** (mirrors Bedrock AgentCore): `Session` (whole conversation) → `Trace` (one user turn) → `Span` (one tool call). All evaluators target exactly one level.
- **Two modes**: `EvalMode.HEURISTIC` (default, offline, rule-based) and `EvalMode.LLM` (uses HF Inference API, requires `HF_TOKEN` or `hf_token` in the request).
- **`SessionTracer`** in `src/wrapper.py` is the integration point for external agents — both as a context manager and a `@trace_agent` decorator. It emits the same JSON schema that `parser.parse_trace()` consumes, so round-tripping is lossless.
- **Adding a new evaluator**: see the contract above. After registering, it is automatically exposed via the `/evaluators` endpoint and selectable in the Gradio UI (which reads from `ALL_EVALUATORS`).

## Do / Don't

- **Do** keep changes scoped to a single concern (one evaluator, one new endpoint, one chart type).
- **Do** preserve the heuristic + LLM dual-path pattern in evaluators.
- **Do** add the new evaluator to `ALL_EVALUATORS` **and** the appropriate level list.
- **Don't** add comments, docstrings to private helpers, or docstrings to functions that don't already have them — match existing density.
- **Don't** introduce `pydantic` models for internal data — use `dataclass`. (Pydantic is used only in `api.py` for HTTP request/response schemas.)
- **Don't** commit `__pycache__/`, `*.pyc`, or `dataset/golden_dataset.jsonl` (it's a generated artifact).
