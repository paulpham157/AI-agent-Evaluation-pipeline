#!/usr/bin/env python3
"""
Golden Dataset Generator — Tech Interview Domains
==================================================
Uses NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 to generate golden (input, expected_output)
pairs for evaluating AI agents conducting **tech job interviews**.

Domains:
  python_backend       — Python, FastAPI, async, OOP
  system_design        — distributed systems, scalability, architecture
  dsa                  — data structures, algorithms, LeetCode-style
  database             — SQL, NoSQL, indexing, schema design
  devops_cloud         — Docker, Kubernetes, CI/CD, AWS/GCP
  machine_learning     — ML concepts, model evaluation, debugging
  javascript_frontend  — React, TypeScript, Node.js, browser APIs
  behavioral_tech      — STAR method behavioral questions for tech roles

Usage:
    python scripts/generate_golden_dataset.py
    python scripts/generate_golden_dataset.py --domains python_backend system_design
    python scripts/generate_golden_dataset.py --records-per-domain 3
    python scripts/generate_golden_dataset.py --dry-run
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
DATASET_REPO = os.getenv("DATASET_REPO", "build-small-hackathon/agent-eval-golden-dataset")
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "dataset/golden_dataset.jsonl"))

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


# ── Core logic (reused by app.py) ────────────────────────────────────────────


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


def call_model(client, template: dict) -> Optional[dict]:
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
                "generated_by": GENERATOR_MODEL,
                "created_at": str(date.today()),
                "tags": [template["domain"], template["difficulty"]],
            },
        }
    except Exception:
        return None


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


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
    )
    parser.add_argument("--records-per-domain", type=int, default=5)
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument(
        "--upload", action="store_true", help="Auto-upload to HF after generation"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    templates = build_templates(args.domains, args.records_per_domain)
    logger.info("🎯 %d records  |  model: %s", len(templates), GENERATOR_MODEL)

    if args.dry_run:
        for t in templates:
            logger.info("  [%s] %s (%s) — %s...", t['domain'], t['id'], t['difficulty'], t['situation'][:60])
        return

    from huggingface_hub import InferenceClient

    client = InferenceClient(model=GENERATOR_MODEL)

    records, failed = [], []

    # Resume support
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["id"])
                records.append(r)
        logger.info("  Resuming — %d records already done", len(records))
        templates = [t for t in templates if t["id"] not in existing_ids]

    with open(output_path, "a", encoding="utf-8") as f:
        for t in templates:
            logger.info("  %s (%s/%s)... ", t['id'], t['domain'], t['difficulty'])
            rec = call_model(client, t)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                records.append(rec)
                logger.info("✓")
            else:
                failed.append(t["id"])
                logger.warning("✗ parse failed")
            time.sleep(0.4)

    logger.info("✅ %d generated  |  ✗ %d failed", len(records), len(failed))
    domains = Counter(r["domain"] for r in records)
    for d, c in sorted(domains.items()):
        logger.info("  %s: %d", d, c)

    if args.upload and records:
        logger.info("📤 Uploading to HF...")
        upload_to_hf(output_path)
        logger.info("✓ Uploaded → https://huggingface.co/datasets/%s", DATASET_REPO)


if __name__ == "__main__":
    main()
