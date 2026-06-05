#!/usr/bin/env python3
"""
Golden Dataset Generator
========================
Uses NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 to generate golden (input, expected_output)
pairs for evaluating AI agents.

Each record contains:
  scenario   — user_goal + system_prompt + initial_message (what to give the agent)
  ground_truth — expected_response + expected_trajectory + assertions (what to check against)

Usage:
    python scripts/generate_golden_dataset.py
    python scripts/generate_golden_dataset.py --limit 5   # generate only 5 records
    python scripts/generate_golden_dataset.py --dry-run   # print prompts, no API calls
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

GENERATOR_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OUTPUT_FILE = Path(__file__).parent.parent / "dataset" / "golden_dataset.jsonl"

# ── Scenario templates ────────────────────────────────────────────────────────
# Each template drives one generation call to Nemotron.

SCENARIO_TEMPLATES = [
    # ── TRAVEL ──────────────────────────────────────────────────────────────
    {
        "id": "travel_001",
        "domain": "travel",
        "difficulty": "easy",
        "agent_role": "travel assistant with tools: search_flights, get_prices",
        "situation": "User wants to find the cheapest flight from Hanoi to Ho Chi Minh City next weekend.",
        "has_tools": True,
    },
    {
        "id": "travel_002",
        "domain": "travel",
        "difficulty": "medium",
        "agent_role": "travel assistant with tools: search_flights, book_flight, check_baggage_policy",
        "situation": "User needs to book a round-trip business class flight NYC→Tokyo for a conference, with specific dates and a budget limit.",
        "has_tools": True,
    },
    {
        "id": "travel_003",
        "domain": "travel",
        "difficulty": "hard",
        "agent_role": "travel assistant with tools: search_flights, search_hotels, check_visa_requirements, book_flight, book_hotel",
        "situation": "User is planning a 2-week trip to Japan in cherry blossom season. They need flights, hotels near train stations, and visa info for a US passport holder.",
        "has_tools": True,
    },
    {
        "id": "travel_004",
        "domain": "travel",
        "difficulty": "easy",
        "agent_role": "travel assistant (no tools, knowledge-based)",
        "situation": "User asks for packing tips for a beach vacation in Thailand in July.",
        "has_tools": False,
    },
    {
        "id": "travel_005",
        "domain": "travel",
        "difficulty": "medium",
        "agent_role": "travel assistant with tools: search_hotels, check_availability, make_reservation",
        "situation": "User wants to book a pet-friendly hotel in Paris for 3 nights, needs to bring a medium-sized dog.",
        "has_tools": True,
    },
    # ── RESTAURANT ───────────────────────────────────────────────────────────
    {
        "id": "restaurant_001",
        "domain": "restaurant",
        "difficulty": "easy",
        "agent_role": "restaurant concierge with tools: search_restaurants, check_availability",
        "situation": "User wants a good sushi restaurant in downtown San Francisco open tonight.",
        "has_tools": True,
    },
    {
        "id": "restaurant_002",
        "domain": "restaurant",
        "difficulty": "medium",
        "agent_role": "restaurant concierge with tools: search_restaurants, check_availability, make_reservation",
        "situation": "User wants to book a romantic Italian restaurant for an anniversary dinner for 2 in NYC. Budget is $$$ and they need a table with a view.",
        "has_tools": True,
    },
    {
        "id": "restaurant_003",
        "domain": "restaurant",
        "difficulty": "hard",
        "agent_role": "restaurant concierge with tools: search_restaurants, check_availability, make_reservation, get_menu",
        "situation": "User is organizing a group dinner for 12 people with mixed dietary restrictions: 3 vegans, 2 gluten-free, 1 nut allergy. Needs a restaurant that can accommodate all.",
        "has_tools": True,
    },
    {
        "id": "restaurant_004",
        "domain": "restaurant",
        "difficulty": "easy",
        "agent_role": "restaurant concierge (knowledge-based, no tools)",
        "situation": "User asks what are the must-try dishes in Vietnamese cuisine for a first-time visitor.",
        "has_tools": False,
    },
    {
        "id": "restaurant_005",
        "domain": "restaurant",
        "difficulty": "medium",
        "agent_role": "restaurant concierge with tools: search_restaurants, make_reservation",
        "situation": "User wants a business lunch venue with private dining room in London, near Bank station. Party of 8, needs AV equipment.",
        "has_tools": True,
    },
    # ── CUSTOMER SUPPORT ─────────────────────────────────────────────────────
    {
        "id": "support_001",
        "domain": "customer_support",
        "difficulty": "easy",
        "agent_role": "customer support agent with tools: lookup_order, get_refund_policy",
        "situation": "Customer received damaged product (headphones) and wants to know the return process.",
        "has_tools": True,
    },
    {
        "id": "support_002",
        "domain": "customer_support",
        "difficulty": "medium",
        "agent_role": "customer support agent with tools: lookup_order, initiate_refund, track_shipment",
        "situation": "Customer's package was marked delivered but never arrived. They need a replacement or refund within 2 days for a birthday gift.",
        "has_tools": True,
    },
    {
        "id": "support_003",
        "domain": "customer_support",
        "difficulty": "hard",
        "agent_role": "customer support agent with tools: lookup_order, lookup_account, escalate_ticket, initiate_refund, apply_coupon",
        "situation": "Long-time customer is threatening to cancel their premium subscription after 3 issues in the past month: a failed delivery, an overcharge, and a broken item. Needs to be retained.",
        "has_tools": True,
    },
    {
        "id": "support_004",
        "domain": "customer_support",
        "difficulty": "easy",
        "agent_role": "customer support agent (knowledge-based)",
        "situation": "Customer wants to know how to reset their password and enable two-factor authentication.",
        "has_tools": False,
    },
    {
        "id": "support_005",
        "domain": "customer_support",
        "difficulty": "medium",
        "agent_role": "customer support agent with tools: check_warranty, create_repair_ticket, schedule_technician",
        "situation": "Customer's 14-month-old washing machine (1-year warranty) stopped working. They want a repair or replacement.",
        "has_tools": True,
    },
    # ── TECHNICAL QA / CODING ────────────────────────────────────────────────
    {
        "id": "tech_001",
        "domain": "technical_qa",
        "difficulty": "easy",
        "agent_role": "coding assistant (no tools, knowledge-based)",
        "situation": "Junior developer asks how to reverse a string in Python and what are 2-3 different ways to do it.",
        "has_tools": False,
    },
    {
        "id": "tech_002",
        "domain": "technical_qa",
        "difficulty": "medium",
        "agent_role": "coding assistant with tools: run_code, search_docs",
        "situation": "Developer has a bug: their async Python function using asyncio is blocking the event loop. They share a code snippet and need it fixed.",
        "has_tools": True,
    },
    {
        "id": "tech_003",
        "domain": "technical_qa",
        "difficulty": "hard",
        "agent_role": "senior software architect (knowledge-based)",
        "situation": "Engineer asks how to design a distributed rate limiter for a microservices API that handles 100k requests/second with Redis.",
        "has_tools": False,
    },
    {
        "id": "tech_004",
        "domain": "technical_qa",
        "difficulty": "medium",
        "agent_role": "DevOps assistant with tools: run_command, check_logs, search_docs",
        "situation": "User's Docker container keeps restarting. They need help diagnosing the issue from the logs and fixing the Dockerfile.",
        "has_tools": True,
    },
    {
        "id": "tech_005",
        "domain": "technical_qa",
        "difficulty": "easy",
        "agent_role": "SQL tutor (knowledge-based)",
        "situation": "Data analyst asks for help writing a SQL query to find the top 5 customers by revenue in the last 30 days, with their total orders count.",
        "has_tools": False,
    },
    # ── RESEARCH / RAG ───────────────────────────────────────────────────────
    {
        "id": "research_001",
        "domain": "research",
        "difficulty": "easy",
        "agent_role": "research assistant with tools: web_search, summarize",
        "situation": "Student asks for a summary of the key differences between supervised, unsupervised, and reinforcement learning.",
        "has_tools": True,
    },
    {
        "id": "research_002",
        "domain": "research",
        "difficulty": "medium",
        "agent_role": "research assistant with tools: search_papers, read_paper, synthesize",
        "situation": "Researcher wants a comparison of the latest LLM evaluation benchmarks (MMLU, HellaSwag, HumanEval) and their limitations.",
        "has_tools": True,
    },
    {
        "id": "research_003",
        "domain": "research",
        "difficulty": "hard",
        "agent_role": "market research analyst with tools: search_web, get_financial_data, analyze_trends",
        "situation": "Startup founder needs a competitive analysis of the top 5 AI coding assistants (GitHub Copilot, Cursor, etc.) including pricing, features, and market share.",
        "has_tools": True,
    },
    {
        "id": "research_004",
        "domain": "research",
        "difficulty": "easy",
        "agent_role": "research assistant (knowledge-based)",
        "situation": "User asks to explain the concept of RAG (Retrieval Augmented Generation) in simple terms with a real-world example.",
        "has_tools": False,
    },
    {
        "id": "research_005",
        "domain": "research",
        "difficulty": "medium",
        "agent_role": "research assistant with tools: search_web, read_article, fact_check",
        "situation": "Journalist needs to fact-check 3 specific claims about AI job displacement statistics before publishing an article.",
        "has_tools": True,
    },
    # ── MOCK INTERVIEW ───────────────────────────────────────────────────────
    {
        "id": "interview_001",
        "domain": "mock_interview",
        "difficulty": "easy",
        "agent_role": "technical interviewer conducting a Python fundamentals screen",
        "situation": "Candidate is a junior developer applying for a backend role. Interviewer should assess Python basics, OOP, and problem-solving.",
        "has_tools": False,
    },
    {
        "id": "interview_002",
        "domain": "mock_interview",
        "difficulty": "medium",
        "agent_role": "system design interviewer with tools: draw_diagram, evaluate_answer",
        "situation": "Senior engineer candidate is asked to design a URL shortener service (like bit.ly) that handles 1 billion URLs. Needs scalability discussion.",
        "has_tools": True,
    },
    {
        "id": "interview_003",
        "domain": "mock_interview",
        "difficulty": "hard",
        "agent_role": "behavioral interviewer assessing leadership and conflict resolution",
        "situation": "Candidate for engineering manager role. Interviewer uses STAR method to probe for examples of managing underperforming team members.",
        "has_tools": False,
    },
    {
        "id": "interview_004",
        "domain": "mock_interview",
        "difficulty": "medium",
        "agent_role": "ML interviewer with tools: run_code, evaluate_model",
        "situation": "ML engineer candidate is asked to explain, then implement, a simple gradient descent optimizer and discuss when to use Adam vs SGD.",
        "has_tools": True,
    },
    {
        "id": "interview_005",
        "domain": "mock_interview",
        "difficulty": "hard",
        "agent_role": "comprehensive technical interviewer (DSA + system design)",
        "situation": "FAANG-level interview: candidate must solve a LeetCode hard problem (LRU cache implementation) AND design a distributed cache after.",
        "has_tools": False,
    },
]

# ── Prompt template ──────────────────────────────────────────────────────────

GENERATION_PROMPT = """\
You are building a golden evaluation benchmark for AI agents.

Create ONE evaluation record for the following scenario:

Agent Role: {agent_role}
Situation: {situation}
Has Tool Calls: {has_tools}
Difficulty: {difficulty}

Generate a JSON record with EXACTLY this structure:
{{
  "user_goal": "<clear one-sentence goal the user wants to achieve>",
  "system_prompt": "<the system prompt that would be given to the agent, 2-4 sentences>",
  "initial_message": "<the user's first message to start the conversation>",
  "expected_response": "<what an ideal final agent response looks like, 3-6 sentences>",
  "expected_trajectory": {trajectory_hint},
  "assertions": [
    "<specific, verifiable assertion 1>",
    "<specific, verifiable assertion 2>",
    "<specific, verifiable assertion 3>"
  ]
}}

Rules:
- expected_response must be what a PERFECT agent would say as its final message
- assertions must be specific and checkable (not vague)
- if has_tools is true, expected_trajectory must list tool names in order
- if has_tools is false, expected_trajectory must be []
- Output ONLY the JSON, no markdown, no explanation
"""

# ── Generator ────────────────────────────────────────────────────────────────


def build_prompt(template: dict) -> str:
    trajectory_hint = (
        '["tool_name_1", "tool_name_2"]' if template["has_tools"] else "[]"
    )
    return GENERATION_PROMPT.format(
        agent_role=template["agent_role"],
        situation=template["situation"],
        has_tools=str(template["has_tools"]).lower(),
        difficulty=template["difficulty"],
        trajectory_hint=trajectory_hint,
    )


def parse_json_from_output(text: str) -> Optional[dict]:
    """Extract JSON from model output, handling markdown code blocks."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find the outermost JSON object
    for match in reversed(list(re.finditer(r"\{[\s\S]{50,}\}", text))):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    # Last attempt: try the whole string
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_record(data: dict) -> bool:
    required = [
        "user_goal",
        "system_prompt",
        "initial_message",
        "expected_response",
        "expected_trajectory",
        "assertions",
    ]
    return all(k in data for k in required) and isinstance(data.get("assertions"), list)


def generate_record(client, template: dict, dry_run: bool = False) -> Optional[dict]:
    prompt = build_prompt(template)

    if dry_run:
        print(f"\n{'=' * 60}")
        print(f"[DRY RUN] {template['id']}")
        print(prompt[:300], "...")
        return None

    print(
        f"  Generating {template['id']} ({template['domain']} / {template['difficulty']})...",
        end=" ",
        flush=True,
    )
    try:
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise benchmark dataset creator. Output ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.75,
        )
        raw = resp.choices[0].message.content or ""
        data = parse_json_from_output(raw)
        if not data or not validate_record(data):
            print("⚠ parse failed")
            return None

        record = {
            "id": template["id"],
            "domain": template["domain"],
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
        print("✓")
        return record
    except Exception as e:
        print(f"✗ error: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Max records to generate"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print prompts without API calls"
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_FILE), help="Output JSONL file path"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    templates = SCENARIO_TEMPLATES
    if args.limit:
        templates = templates[: args.limit]

    print(f"🎯 Generating {len(templates)} golden records with {GENERATOR_MODEL}")
    print(f"📁 Output: {output_path}\n")

    if args.dry_run:
        for t in templates:
            generate_record(None, t, dry_run=True)
        return

    # Init client
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=GENERATOR_MODEL)

    records = []
    failed = []

    # Load existing records to allow resuming
    if output_path.exists():
        existing_ids = set()
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["id"])
                records.append(r)
        print(f"  Resuming — {len(records)} records already generated\n")
        templates = [t for t in templates if t["id"] not in existing_ids]

    with open(output_path, "a", encoding="utf-8") as f:
        for template in templates:
            record = generate_record(client, template)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                records.append(record)
            else:
                failed.append(template["id"])
            time.sleep(0.5)  # gentle rate limiting

    print(f"\n{'=' * 50}")
    print(f"✅ Generated: {len(records)} records")
    if failed:
        print(f"⚠ Failed:    {len(failed)} records: {failed}")
    print(f"📁 Saved to: {output_path}")

    # Print domain breakdown
    from collections import Counter

    domains = Counter(r["domain"] for r in records)
    print("\nDomain breakdown:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain:20s}: {count}")


if __name__ == "__main__":
    main()
