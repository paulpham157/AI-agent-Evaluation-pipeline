"""
Golden Dataset Generator — Shared logic
========================================
Reused by both `scripts/generate_golden_dataset.py` (CLI) and `app.py` (Gradio UI).

Supports two backends:
  - "inference" — HF Inference API (default, no local GPU needed)
  - "llama-cpp" — llama.cpp via llama-cpp-python (requires GPU)
"""

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ─── Config (env-driven) ──────────────────────────────────────────────────────
DATASET_REPO = os.getenv("DATASET_REPO", "build-small-hackathon/agent-eval-golden-dataset")
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "dataset/golden_dataset.jsonl"))

LLAMA_MODEL_REPO = os.getenv("LLAMA_MODEL_REPO", "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF")
LLAMA_MODEL_FILE = os.getenv("LLAMA_MODEL_FILE", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf")
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "16384"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))
LLAMA_N_BATCH = int(os.getenv("LLAMA_N_BATCH", "512"))

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

# ── Tech Interview Scenario Templates ────────────────────────────────────────

SCENARIOS = {
    "python_backend": [
        {
            "id": "python_001",
            "difficulty": "easy",
            "situation": "Junior dev asks how to reverse a string in Python — explain 3 different approaches with time complexity.",
            "has_tools": False,
        },
        {
            "id": "python_002",
            "difficulty": "medium",
            "situation": "Candidate must implement a thread-safe LRU cache class in Python using OrderedDict and explain the design choices.",
            "has_tools": False,
        },
        {
            "id": "python_003",
            "difficulty": "medium",
            "situation": "Developer shares code where an async FastAPI endpoint blocks the event loop. Interviewer must identify and fix the issue.",
            "has_tools": True,
        },
        {
            "id": "python_004",
            "difficulty": "hard",
            "situation": "Design a Python decorator that implements rate limiting (N calls per second) with thread safety and expiry.",
            "has_tools": False,
        },
        {
            "id": "python_005",
            "difficulty": "easy",
            "situation": "Explain Python's GIL, when it matters, and demonstrate with an I/O-bound vs CPU-bound example.",
            "has_tools": False,
        },
    ],
    "system_design": [
        {
            "id": "sysdesign_001",
            "difficulty": "easy",
            "situation": "Design a URL shortener (like bit.ly) that handles 100M URLs. Cover storage, hashing, and redirect flow.",
            "has_tools": False,
        },
        {
            "id": "sysdesign_002",
            "difficulty": "medium",
            "situation": "Design a real-time leaderboard for a mobile game with 10M daily active users. Must support top-100 and user rank queries.",
            "has_tools": False,
        },
        {
            "id": "sysdesign_003",
            "difficulty": "hard",
            "situation": "Design a distributed rate limiter for a public API handling 100k requests/second across 5 data centers.",
            "has_tools": False,
        },
        {
            "id": "sysdesign_004",
            "difficulty": "medium",
            "situation": "Design a notification service (push, email, SMS) for 50M users with guaranteed delivery and deduplication.",
            "has_tools": False,
        },
        {
            "id": "sysdesign_005",
            "difficulty": "hard",
            "situation": "Design Twitter's home feed ranking system — real-time ingestion, personalized ranking, fan-out strategies.",
            "has_tools": False,
        },
    ],
    "dsa": [
        {
            "id": "dsa_001",
            "difficulty": "easy",
            "situation": "Find all duplicates in an integer array — explain and compare O(n²), O(n log n), and O(n) approaches.",
            "has_tools": False,
        },
        {
            "id": "dsa_002",
            "difficulty": "medium",
            "situation": "Implement a stack that supports push, pop, and getMin() all in O(1) time and space. Explain the invariant.",
            "has_tools": False,
        },
        {
            "id": "dsa_003",
            "difficulty": "hard",
            "situation": "Implement LRU Cache with get() and put() in O(1). Candidate must use doubly linked list + hashmap.",
            "has_tools": False,
        },
        {
            "id": "dsa_004",
            "difficulty": "medium",
            "situation": "Given a binary tree, write iterative in-order traversal without recursion. Explain the stack-based approach.",
            "has_tools": False,
        },
        {
            "id": "dsa_005",
            "difficulty": "easy",
            "situation": "Explain when to use BFS vs DFS with concrete examples. Implement BFS shortest path on an unweighted graph.",
            "has_tools": False,
        },
    ],
    "database": [
        {
            "id": "db_001",
            "difficulty": "easy",
            "situation": "Write SQL to find the top 5 customers by total revenue in the last 30 days, including their order count.",
            "has_tools": False,
        },
        {
            "id": "db_002",
            "difficulty": "medium",
            "situation": "Optimize a slow query joining orders, customers, and products tables that takes 8 seconds on 10M rows.",
            "has_tools": True,
        },
        {
            "id": "db_003",
            "difficulty": "hard",
            "situation": "Design a database schema for an e-commerce system supporting products, variants, inventory, orders, and reviews.",
            "has_tools": False,
        },
        {
            "id": "db_004",
            "difficulty": "easy",
            "situation": "Explain ACID properties with real-world examples of each. When would you relax consistency for availability?",
            "has_tools": False,
        },
        {
            "id": "db_005",
            "difficulty": "medium",
            "situation": "When to choose PostgreSQL vs MongoDB vs Redis — walk through 3 concrete scenarios justifying each choice.",
            "has_tools": False,
        },
    ],
    "devops_cloud": [
        {
            "id": "devops_001",
            "difficulty": "easy",
            "situation": "Write a production-ready Dockerfile for a Python FastAPI app with multi-stage build, non-root user, and health check.",
            "has_tools": False,
        },
        {
            "id": "devops_002",
            "difficulty": "medium",
            "situation": "A Kubernetes pod is in CrashLoopBackOff. Walk through the diagnostic steps using kubectl commands.",
            "has_tools": True,
        },
        {
            "id": "devops_003",
            "difficulty": "hard",
            "situation": "Design a CI/CD pipeline for a microservices app with 20 services — blue/green deploy, rollback, and canary releases.",
            "has_tools": False,
        },
        {
            "id": "devops_004",
            "difficulty": "easy",
            "situation": "Explain the difference between Docker containers and VMs. When would you choose one over the other?",
            "has_tools": False,
        },
        {
            "id": "devops_005",
            "difficulty": "medium",
            "situation": "Set up monitoring and alerting for a web service — define key SLIs/SLOs and implement with Prometheus + Grafana.",
            "has_tools": False,
        },
    ],
    "machine_learning": [
        {
            "id": "ml_001",
            "difficulty": "easy",
            "situation": "Explain overfitting, how to detect it from learning curves, and describe 4 regularization techniques.",
            "has_tools": False,
        },
        {
            "id": "ml_002",
            "difficulty": "medium",
            "situation": "A model has 99% training accuracy but 60% validation accuracy. Debug the issue and propose fixes.",
            "has_tools": True,
        },
        {
            "id": "ml_003",
            "difficulty": "hard",
            "situation": "Design an ML pipeline for real-time fraud detection: feature engineering, model selection, latency constraints, retraining.",
            "has_tools": False,
        },
        {
            "id": "ml_004",
            "difficulty": "easy",
            "situation": "Explain the bias-variance tradeoff and give examples of high-bias vs high-variance models.",
            "has_tools": False,
        },
        {
            "id": "ml_005",
            "difficulty": "medium",
            "situation": "Compare when to use gradient boosting (XGBoost) vs neural networks for tabular data. Walk through decision criteria.",
            "has_tools": False,
        },
    ],
    "javascript_frontend": [
        {
            "id": "js_001",
            "difficulty": "easy",
            "situation": "Explain JavaScript's event loop, call stack, and microtask queue. Predict output of a Promise + setTimeout snippet.",
            "has_tools": False,
        },
        {
            "id": "js_002",
            "difficulty": "medium",
            "situation": "A React component re-renders too frequently causing performance issues. Diagnose and fix using useMemo/useCallback.",
            "has_tools": True,
        },
        {
            "id": "js_003",
            "difficulty": "hard",
            "situation": "Design the frontend architecture for a large SPA — code splitting, state management, micro-frontends decision.",
            "has_tools": False,
        },
        {
            "id": "js_004",
            "difficulty": "easy",
            "situation": "Explain JavaScript closures with 3 practical use cases: counter, memoization, and partial application.",
            "has_tools": False,
        },
        {
            "id": "js_005",
            "difficulty": "medium",
            "situation": "Implement a debounce function in TypeScript with generics. Explain use cases vs throttle.",
            "has_tools": False,
        },
    ],
    "behavioral_tech": [
        {
            "id": "behavioral_001",
            "difficulty": "easy",
            "situation": "Using STAR method: 'Tell me about a time you had a technical disagreement with a senior engineer. How did it resolve?'",
            "has_tools": False,
        },
        {
            "id": "behavioral_002",
            "difficulty": "medium",
            "situation": "Using STAR method: 'Describe how you led a complex technical project under a tight deadline with incomplete requirements.'",
            "has_tools": False,
        },
        {
            "id": "behavioral_003",
            "difficulty": "hard",
            "situation": "Using STAR method: 'Tell me about a critical production incident you caused. Walk through your response and what you learned.'",
            "has_tools": False,
        },
        {
            "id": "behavioral_004",
            "difficulty": "medium",
            "situation": "Using STAR method: 'How have you mentored a struggling junior engineer? What was your approach and the outcome?'",
            "has_tools": False,
        },
        {
            "id": "behavioral_005",
            "difficulty": "easy",
            "situation": "Using STAR method: 'Describe your code review process. How do you give constructive feedback on bad code?'",
            "has_tools": False,
        },
    ],
}

DOMAIN_LABELS = {
    "python_backend": "Python & Backend",
    "system_design": "System Design",
    "dsa": "Data Structures & Algorithms",
    "database": "Database & SQL",
    "devops_cloud": "DevOps & Cloud",
    "machine_learning": "Machine Learning",
    "javascript_frontend": "JavaScript & Frontend",
    "behavioral_tech": "Behavioral (Tech)",
}

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT = """\
You are building a golden benchmark dataset for evaluating AI tech interviewers.

Create ONE evaluation record for this tech interview scenario:

Domain: {domain_label}
Situation: {situation}
Interviewer has tools: {has_tools}
Difficulty: {difficulty}

Output a JSON object with EXACTLY these fields:
{{
  "user_goal": "<one-sentence goal of what the interviewer should achieve in this session>",
  "system_prompt": "<2-3 sentence system prompt for the AI interviewer agent>",
  "initial_message": "<candidate's opening message to start the interview turn>",
  "expected_response": "<ideal interviewer response — clear, pedagogical, technically accurate, 3-5 sentences>",
  "expected_trajectory": {trajectory},
  "assertions": [
    "<specific verifiable assertion 1 about what the ideal response must contain>",
    "<specific verifiable assertion 2>",
    "<specific verifiable assertion 3>"
  ]
}}

Rules:
- expected_response is what a PERFECT interviewer would say
- assertions must be concrete and checkable (e.g. "Response explains time complexity", not "Response is good")
- Output ONLY the JSON, no markdown fences, no extra text
"""


# ── Core logic ────────────────────────────────────────────────────────────────


def build_templates(domains: list[str], records_per_domain: int) -> list[dict]:
    """Return flattened list of templates for selected domains."""
    templates = []
    for domain in domains:
        domain_scenarios = SCENARIOS.get(domain, [])[:records_per_domain]
        for s in domain_scenarios:
            templates.append({**s, "domain": domain})
    return templates


def make_prompt(template: dict) -> str:
    traj = '["tool_name_1", "tool_name_2"]' if template["has_tools"] else "[]"
    return PROMPT.format(
        domain_label=DOMAIN_LABELS.get(template["domain"], template["domain"]),
        situation=template["situation"],
        has_tools=str(template["has_tools"]).lower(),
        difficulty=template["difficulty"],
        trajectory=traj,
    )


def parse_output(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    for m in reversed(list(re.finditer(r"\{[\s\S]{40,}\}", text))):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate(data: dict) -> bool:
    required = [
        "user_goal",
        "system_prompt",
        "initial_message",
        "expected_response",
        "expected_trajectory",
        "assertions",
    ]
    return (
        all(k in data for k in required)
        and isinstance(data.get("assertions"), list)
        and len(data["assertions"]) >= 1
    )


# ─── Backend: llama.cpp Client ─────────────────────────────────────


class LlamaCppClient:
    """llama.cpp client for local GPU inference via llama-cpp-python.

    Downloads GGUF model from HF Hub on first use (lazy init).
    Chat-compatible interface with the same .chat_completion() signature
    as HF InferenceClient, so callers can swap backends transparently.
    """
    """llama.cpp client for local GPU inference via llama-cpp-python."""

    def __init__(
        self,
        model_repo: str = LLAMA_MODEL_REPO,
        model_file: str = LLAMA_MODEL_FILE,
        n_ctx: int = LLAMA_N_CTX,
        n_gpu_layers: int = LLAMA_N_GPU_LAYERS,
        n_batch: int = LLAMA_N_BATCH,
    ):
        self.model_repo = model_repo
        self.model_file = model_file
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self._llm = None

    def _init_llm(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python") from e

        from huggingface_hub import hf_hub_download

        logger.info("Downloading model %s/%s ...", self.model_repo, self.model_file)
        model_path = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_file,
        )
        logger.info("Loading model from %s (n_ctx=%d, n_gpu_layers=%d, n_batch=%d) ...",
                     model_path, self.n_ctx, self.n_gpu_layers, self.n_batch)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            verbose=False,
        )
        logger.info("llama.cpp model loaded successfully")

    def chat_completion(self, messages, max_tokens=1200, temperature=0.7, top_p=0.95):
        self._init_llm()

        prompt = self._messages_to_prompt(messages)

        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>"],
        )

        generated_text = output["choices"][0]["text"] if output.get("choices") else ""

        class Choice:
            def __init__(self, text):
                self.message = type('Message', (), {'content': text})()

        class Response:
            def __init__(self, text):
                self.choices = [Choice(text)]

        return Response(generated_text)

    def _messages_to_prompt(self, messages) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


# ─── Backend callers ────────────────────────────────────────────────


def call_model_llamacpp(client: LlamaCppClient, template: dict, model_name: str = None) -> Optional[dict]:
    """Call model via local llama.cpp."""
    if model_name is None:
        model_name = f"{LLAMA_MODEL_REPO}/{LLAMA_MODEL_FILE}"
    try:
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise benchmark creator. Output ONLY valid JSON.",
                },
                {"role": "user", "content": make_prompt(template)},
            ],
            max_tokens=1200,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content or ""
        data = parse_output(raw)
        if not data or not validate(data):
            return None
        return _build_result(template, data, model_name=model_name)
    except Exception as e:
        logger.warning("llama.cpp call failed for %s: %s", template["id"], e)
        return None


def call_model_inference(model_name: str, template: dict, hf_token: str = None) -> Optional[dict]:
    """Call model via HF Inference API."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_name, token=hf_token)
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise benchmark creator. Output ONLY valid JSON.",
                },
                {"role": "user", "content": make_prompt(template)},
            ],
            max_tokens=1200,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content or ""
        data = parse_output(raw)
        if not data or not validate(data):
            return None
        return _build_result(template, data, model_name=model_name)
    except Exception as e:
        logger.warning("Inference API call failed for %s: %s", template["id"], e)
        return None


def _build_result(template: dict, data: dict, model_name: str = None) -> dict:
    """Build result dict from parsed data."""
    if model_name is None:
        model_name = f"{LLAMA_MODEL_REPO}/{LLAMA_MODEL_FILE}"
    return {
        "id": template["id"],
        "domain": template["domain"],
        "domain_label": DOMAIN_LABELS.get(template["domain"], template["domain"]),
        "difficulty": template["difficulty"],
        "has_tools": template["has_tools"],
        "scenario": {
            "user_goal": data["user_goal"],
            "system_prompt": data["system_prompt"],
            "initial_message": data["initial_message"],
        },
        "ground_truth": {
            "expected_response": data["expected_response"],
            "expected_trajectory": data.get("expected_trajectory", []),
            "assertions": data["assertions"],
        },
        "metadata": {
            "generated_by": model_name,
            "created_at": str(date.today()),
            "tags": [template["domain"], template["difficulty"]],
        },
    }


def upload_to_hf(output_path: Path, hf_token: str = None):
    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo="data/golden_dataset.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"data: generated golden dataset ({date.today()})",
    )


# ─── Orchestrator ──────────────────────────────────────────────────


def generate_dataset(
    domains: list[str],
    records_per_domain: int,
    backend: str = "inference",
    model_name: str = None,
    client: LlamaCppClient = None,
    output_path: str = None,
    upload: bool = False,
    hf_token: str = None,
    progress_callback: Callable = None,
) -> tuple:
    """Generate golden dataset records.

    Parameters
    ----------
    domains : list of domain keys.
    records_per_domain : int.
    backend : "inference" | "llama-cpp".
    model_name : override model name (repo_id for inference, repo for llama-cpp).
    client : pre-loaded LlamaCppClient (if None, a new one is created lazily).
    output_path : path to save JSONL.
    upload : upload to HF dataset repo after generation.
    hf_token : HF token for Inference API / upload.
    progress_callback : fn(current, total, message) for UI updates.

    Returns
    -------
    (records: list, failed: list[str], log_lines: list[str])
    """
    if output_path is None:
        output_path = str(OUTPUT_FILE)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    log: list[str] = []

    def _log(msg: str):
        logger.info(msg)
        log.append(msg)

    templates = build_templates(domains, records_per_domain)
    total = len(templates)

    if backend == "llama-cpp":
        if model_name:
            repo = model_name
        else:
            repo = LLAMA_MODEL_REPO
        resolved_model = f"{repo}/{LLAMA_MODEL_FILE}" if "/" not in repo or "." not in repo else repo
        _log(f"🎯 {total} records  |  backend: llama.cpp  |  model: {resolved_model}")
        if client is None:
            client = LlamaCppClient(model_repo=repo)
        call_fn = lambda t: call_model_llamacpp(client=client, template=t, model_name=resolved_model)
    else:
        resolved_model = model_name or GENERATOR_MODEL
        _log(f"🎯 {total} records  |  backend: Inference API  |  model: {resolved_model}")
        call_fn = lambda t: call_model_inference(resolved_model, t, hf_token=hf_token)

    if progress_callback:
        progress_callback(0, total, "Starting generation...")

    records, failed = [], []

    # Resume support
    existing_ids = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["id"])
                records.append(r)
        _log(f"  Resuming — {len(records)} records already done")
        templates = [t for t in templates if t["id"] not in existing_ids]

    with open(out, "a", encoding="utf-8") as f:
        for i, t in enumerate(templates):
            msg = f"  {t['id']} ({t['domain']}/{t['difficulty']})..."
            _log(msg)
            if progress_callback:
                progress_callback(i, total, t['id'])

            rec = call_fn(t)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                records.append(rec)
                _log("✓")
            else:
                failed.append(t["id"])
                _log("✗ parse failed")
            time.sleep(0.4)

    _log(f"✅ {len(records)} generated  |  ✗ {len(failed)} failed")
    domains_found = Counter(r["domain"] for r in records)
    for d, c in sorted(domains_found.items()):
        _log(f"  {d}: {c}")

    if upload and records:
        _log("📤 Uploading to HF...")
        upload_to_hf(out, hf_token=hf_token)
        _log(f"✓ Uploaded → https://huggingface.co/datasets/{DATASET_REPO}")

    return records, failed, log
