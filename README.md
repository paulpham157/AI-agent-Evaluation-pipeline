---
title: ai agent evaluation pipeline
emoji: 🧪
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: Evaluate AI agents at Session, Trace & Span levels
tags:
  - evaluation
  - agents
  - llm
  - gradio
  - observability
---

# 🧪 AI Agent Evaluation Pipeline

> Evaluate AI agents at **Session**, **Trace**, and **Span** levels — inspired by [Amazon Bedrock AgentCore Evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html)

## Overview

This tool provides a structured framework for evaluating AI agent conversations using the same three-level hierarchy as Amazon Bedrock AgentCore Evaluations:

```
📦 Session  → Did the agent achieve the user's overall goal?
  └── 🔄 Trace   → Per-turn quality (helpfulness, coherence, relevance...)
        └── 🔧 Span   → Per tool-call accuracy
```

## Features

- **14 built-in evaluators** (1 session + 11 trace + 2 span)
- **Heuristic mode** — works offline, no API key required
- **3 demo traces** (Simple Q&A, Tool Calling, Multi-turn)
- **Ground truth support** — `expected_response`, `expected_trajectory`, `assertions`
- **Visual results** — radar chart, bar chart, heatmap, score cards

## Evaluators

### 📦 Session Level (1)

| Evaluator         | Description                                         |
| ----------------- | --------------------------------------------------- |
| Goal Success Rate | Did the agent fully achieve the user's stated goal? |

### 🔄 Trace Level (11)

| Evaluator               | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| Helpfulness             | Does the response help the user progress toward their goal? |
| Correctness             | Is the response factually correct?                          |
| Coherence               | Is the reasoning logically consistent and well-structured?  |
| Conciseness             | Is the response appropriately concise?                      |
| Faithfulness            | Is the response consistent with conversation history?       |
| Harmfulness             | Does the response contain harmful content?                  |
| Instruction Following   | Does the agent follow its system prompt?                    |
| Response Relevance      | Does the response address what was asked?                   |
| Context Relevance       | Was the retrieved context relevant? (RAG)                   |
| Refusal Appropriateness | Did the agent correctly handle refusals?                    |
| Stereotyping / Bias     | Is there demographic bias in the response?                  |

### 🔧 Span Level (2)

| Evaluator               | Description                            |
| ----------------------- | -------------------------------------- |
| Tool Selection Accuracy | Did the agent choose the right tool?   |
| Tool Parameter Accuracy | Did the agent pass correct parameters? |

## JSON Trace Format

```json
{
  "session_id": "my_session",
  "user_goal": "The user's overall goal for this conversation",
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
          "tool_input": { "param": "value" },
          "tool_output": "Tool result",
          "duration_ms": 250
        }
      ]
    }
  ]
}
```

## Ground Truth Support

Optional reference inputs for more precise evaluation:

- **`expected_response`** — What the final response should look like (enables Correctness scoring)
- **`expected_trajectory`** — Expected tool call sequence (enables TrajectoryMatch scoring)
- **`assertions`** — Natural language assertions about the session (enables GoalSuccessRate scoring)

## Running Locally

```bash
git clone https://github.com/your-org/ai-agent-eval-pipeline
cd ai-agent-eval-pipeline
pip install -r requirements.txt

# Gradio UI
python app.py                     # http://localhost:7860

# REST API
python api.py                     # http://localhost:8000
# or
uvicorn api:app --reload --port 8000
```

## Integration — Zero Changes to Your Agent

### Option 1 — Python Wrapper

```python
from src.wrapper import SessionTracer

with SessionTracer(
    goal="Interview a Python candidate",
    system_prompt="You are a technical interviewer...",
) as tracer:
    for user_msg in conversation:
        # Your agent code — completely unchanged
        response = my_agent.invoke(user_msg)

        # Optional: capture tool calls made during this turn
        span = tracer.new_span()
        span.log_span("search_kb", {"query": user_msg}, kb_result)

        tracer.log_trace(user_msg, response, span)

    report = tracer.evaluate()
    print(f"Overall: {report.overall_score:.0%}")
    tracer.save("traces/session_001.json")
```

### Option 2 — REST API

```bash
# Start the server
python api.py   # → http://localhost:8000

# Evaluate a session
curl -X POST http://localhost:8000/evaluate/quick \
  -H "Content-Type: application/json" \
  -d '{
    "trace": {
      "session_id": "interview_001",
      "user_goal": "Assess Python skills",
      "traces": [
        {
          "trace_id": "t1",
          "user_input": "What is a decorator?",
          "agent_response": "A decorator is a function that wraps another function...",
          "spans": []
        }
      ]
    }
  }'
```

API docs auto-generated at `http://localhost:8000/docs`.

## Architecture

```
app.py                  # Gradio UI entry point
api.py                  # FastAPI REST server
src/
├── models.py           # Session / Trace / Span / EvalScore data classes
├── parser.py           # JSON trace parser
├── evaluators.py       # All 14 evaluators (heuristic + LLM-ready)
├── runner.py           # Evaluation orchestrator
├── visualizer.py       # Plotly charts
└── wrapper.py          # SessionTracer — captures agent conversations
demos/
├── simple_qa.json      # Demo: Simple Q&A
├── tool_calling.json   # Demo: Tool calling
└── multi_turn.json     # Demo: Multi-turn with tools
```

## Roadmap

### ✅ MVP Complete

- [x] **Gradio UI** — 14 evaluators, Session / Trace / Span levels, 3 demo traces
- [x] **Agent Wrapper** (`src/wrapper.py`) — `SessionTracer` + `trace_agent` decorator
- [x] **REST API** (`api.py`) — `POST /evaluate`, `POST /evaluate/quick`, `GET /evaluators`
- [x] **LLM-as-Judge** (`src/llm_judge.py`) — `Qwen/Qwen3.6-27B` via HF Inference API
- [x] **pass@k / pass^k** (`src/reliability.py`) — multi-trial reliability metrics
- [x] **Golden Dataset Generator** — Nemotron-3-Nano-30B, 8 tech interview domains
- [x] **Deployed** — `build-small-hackathon/AI-agent-Evaluation-pipeline`

### 📋 Future (post-MVP)

- [ ] Export results as JSON / CSV
- [ ] Custom evaluator builder (user-defined prompt templates)
- [ ] Dataset management for regression testing
- [ ] Online monitoring mode

## Inspiration

This project is inspired by the architecture and evaluator design of [Amazon Bedrock AgentCore Evaluations](https://aws.amazon.com/blogs/machine-learning/build-reliable-ai-agents-with-amazon-bedrock-agentcore-evaluations/), re-implemented as an open-source Gradio application.

## License

MIT
