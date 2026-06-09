#!/usr/bin/env python3
"""
Golden Dataset Generator — Tech Interview Domains
==================================================
CLI entry point that delegates to src.dataset_generator.

Backends:
  - inference (default) — HF Inference API
  - llama-cpp           — llama-cpp-python (ZeroGPU)

Usage:
    python scripts/generate_golden_dataset.py
    python scripts/generate_golden_dataset.py --domains python_backend system_design
    python scripts/generate_golden_dataset.py --records-per-domain 3
    python scripts/generate_golden_dataset.py --dry-run
    python scripts/generate_golden_dataset.py --backend llama-cpp --dry-run
    python scripts/generate_golden_dataset.py --backend inference --model meta-llama/Llama-3.2-3B-Instruct
    python scripts/generate_golden_dataset.py --upload
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.dataset_generator import (
    SCENARIOS,
    generate_dataset,
    GENERATOR_MODEL,
    LLAMA_MODEL_REPO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
    )
    parser.add_argument("--records-per-domain", type=int, default=5)
    parser.add_argument("--output", default="dataset/golden_dataset.jsonl")
    parser.add_argument(
        "--backend",
        choices=["inference", "llama-cpp"],
        default=os.getenv("GENERATOR_BACKEND", "inference"),
        help="Backend: 'inference' (HF API) or 'llama-cpp' (local GPU)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (HF model ID for inference, GGUF repo for llama-cpp)",
    )
    parser.add_argument(
        "--upload", action="store_true", help="Auto-upload to HF after generation"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from src.dataset_generator import build_templates

    if args.dry_run:
        templates = build_templates(args.domains, args.records_per_domain)
        print(f"🎯 {len(templates)} records (dry-run)")
        for t in templates:
            print(f"  [{t['domain']}] {t['id']} ({t['difficulty']}) — {t['situation'][:60]}...")
        return

    records, failed, log = generate_dataset(
        domains=args.domains,
        records_per_domain=args.records_per_domain,
        backend=args.backend,
        model_name=args.model,
        output_path=args.output,
        upload=args.upload,
        hf_token=os.getenv("HF_TOKEN"),
    )
    for line in log:
        print(line)


if __name__ == "__main__":
    main()
