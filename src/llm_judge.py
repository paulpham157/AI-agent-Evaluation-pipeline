"""
LLM Judge — Qwen/QwQ-32B via HuggingFace Inference API
========================================================
QwQ-32B is a reasoning model: it outputs <think>…</think> before the answer.
This module strips the thinking block and parses the final JSON score.

Usage:
    from src.llm_judge import LLMJudge
    judge = LLMJudge(api_key="hf_...")
    score, explanation = judge.score(prompt)   # score is 0.0–1.0
"""

import json
import os
import re
from typing import Optional, Tuple


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class LLMJudge:
    """
    Thin wrapper around HuggingFace InferenceClient for evaluation scoring.

    Parameters
    ----------
    model_id : str
        HF model repo ID. Defaults to Qwen/QwQ-32B.
    api_key  : str, optional
        HF access token. Falls back to HF_TOKEN env var.
    timeout  : int
        Request timeout in seconds (default 90 — QwQ reasons before answering).
    """

    DEFAULT_MODEL = "Qwen/QwQ-32B"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: int = 90,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("HF_TOKEN", "")
        self.timeout = timeout
        self._client = None

    # ── Public interface ─────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if an API key is set."""
        return bool(self.api_key)

    def score(self, prompt: str, max_tokens: int = 1024) -> Tuple[float, str]:
        """
        Send an evaluation prompt to QwQ-32B.

        Returns
        -------
        (normalized_score, explanation)
            normalized_score : float in [0.0, 1.0]  (input scale 1-5 / 5)
            explanation      : str rationale from the model
        """
        if not self.available:
            return 0.5, "No HF_TOKEN — using heuristic mode instead."
        try:
            client = self._get_client()
            resp = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise AI quality evaluator. "
                            "Think carefully, then output ONLY valid JSON on the last line."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content or ""
            return self._parse(raw)
        except Exception as exc:
            return 0.5, f"LLM judge error ({type(exc).__name__}): {exc}"

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # lazy import

            self._client = InferenceClient(
                model=self.model_id,
                token=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def _parse(self, raw: str) -> Tuple[float, str]:
        """
        Extract (score, explanation) from QwQ output.

        QwQ outputs:
            <think>… long reasoning …</think>
            {"score": 4, "explanation": "…"}
        """
        # 1. Strip <think>…</think> reasoning block
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # 2. Try every JSON-looking substring, last one wins
        for match in reversed(list(re.finditer(r"\{[^{}]{2,800}\}", text, re.DOTALL))):
            try:
                data = json.loads(match.group())
                score_raw = float(
                    data.get("score") or data.get("rating") or data.get("value") or 3
                )
                explanation = str(
                    data.get("explanation")
                    or data.get("reason")
                    or data.get("reasoning")
                    or ""
                ).strip()
                if not explanation:
                    # use surrounding text as explanation
                    explanation = text.replace(match.group(), "").strip()[:300]
                return _clamp(score_raw / 5.0), explanation or "No explanation."
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # 3. Regex fallback
        m = re.search(r"(?:score|rating)\s*[=:]\s*(\d(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            return _clamp(float(m.group(1)) / 5.0), text[:400]

        # 4. Last resort
        return 0.5, f"Could not parse score. Raw output: {text[:250]}"
