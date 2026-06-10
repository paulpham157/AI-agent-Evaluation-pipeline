"""
LLM Judge — Inference API (remote) + Local llama.cpp backend
=============================================================
Two judge backends sharing the same .score(prompt) interface:

  - LLMJudge:      HF Inference API (remote, any HF model)
  - LocalQwenJudge: llama-cpp-python (local, GGUF models)

Usage:
    from src.llm_judge import LLMJudge, LocalQwenJudge

    # Remote judge
    judge = LLMJudge(api_key="hf_...")
    score, explanation = judge.score(prompt)

    # Local judge
    judge = LocalQwenJudge(model_path="/path/to/model.gguf")
    score, explanation = judge.score(prompt)
"""

import json
import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _parse_judge_output(raw: str) -> Tuple[float, str]:
    """
    Parse (score, explanation) from LLM judge output.

    Expected format:
        {"score": 4, "explanation": "..."}
    May include  <think>...</think>  reasoning block (QwQ style).
    """
    # 1. Strip <think>…</think> block
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


# ─── Remote judge: HF Inference API ─────────────────────────────────


class LLMJudge:
    """
    Judge via HuggingFace Inference API (remote).

    Parameters
    ----------
    model_id : str
        HF model repo ID. Defaults to Qwen/Qwen3.6-27B.
    api_key  : str, optional
        HF access token. Falls back to HF_TOKEN env var.
    timeout  : int
        Request timeout in seconds (default 90).
    """

    DEFAULT_MODEL = "Qwen/Qwen3.6-27B"

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

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def score(self, prompt: str, max_tokens: int = 1024) -> Tuple[float, str]:
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
            return _parse_judge_output(raw)
        except Exception as exc:
            return 0.5, f"LLM judge error ({type(exc).__name__}): {exc}"

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # lazy import

            self._client = InferenceClient(
                model=self.model_id,
                token=self.api_key,
                timeout=self.timeout,
            )
        return self._client


# ─── Local judge: llama.cpp GGUF ─────────────────────────────────────


class LocalQwenJudge:
    """
    Judge via local llama-cpp-python (GGUF model).

    Parameters
    ----------
    model_path : str
        Path to GGUF file, or HF repo ID for auto-download (requires model_file).
    model_file : str, optional
        GGUF filename within the HF repo (required if model_path is a repo ID).
    n_ctx : int
        Context length (default 8192).
    n_gpu_layers : int
        GPU layers to offload (-1 = all, default -1).
    n_batch : int
        Batch size (default 512).
    """

    DEFAULT_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    DEFAULT_FILE = "qwen2.5-7b-instruct-q4_k_m.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_file: Optional[str] = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
    ):
        self.model_path = model_path
        self.model_file = model_file
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self._llm = None
        self._load_error = None

    @property
    def available(self) -> bool:
        return self._llm is not None

    def _init_llm(self):
        if self._llm is not None or self._load_error is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as e:
            self._load_error = f"llama-cpp-python not installed ({e})"
            logger.error(self._load_error)
            return

        resolved_path = self.model_path or os.getenv("LOCAL_JUDGE_MODEL_PATH", "")
        resolved_file = self.model_file or os.getenv("LOCAL_JUDGE_MODEL_FILE", "")

        if not resolved_path and not resolved_file:
            resolved_path = self.DEFAULT_REPO
            resolved_file = self.DEFAULT_FILE

        if not resolved_path.endswith(".gguf") and resolved_file:
            from huggingface_hub import hf_hub_download
            try:
                logger.info("Downloading %s/%s ...", resolved_path, resolved_file)
                resolved_path = hf_hub_download(
                    repo_id=resolved_path,
                    filename=resolved_file,
                )
            except Exception as e:
                self._load_error = f"Failed to download model: {e}"
                logger.error(self._load_error)
                return

        if not os.path.isfile(resolved_path):
            self._load_error = f"Model file not found: {resolved_path}"
            logger.error(self._load_error)
            return

        logger.info(
            "Loading local judge %s (n_ctx=%d, n_gpu_layers=%d) ...",
            resolved_path, self.n_ctx, self.n_gpu_layers,
        )
        try:
            self._llm = Llama(
                model_path=resolved_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch,
                verbose=False,
            )
            logger.info("Local judge loaded successfully")
        except Exception as e:
            self._load_error = f"Failed to load model: {e}"
            logger.error(self._load_error)

    def score(self, prompt: str, max_tokens: int = 512) -> Tuple[float, str]:
        self._init_llm()
        if self._llm is None:
            err = self._load_error or "Local judge not available"
            return 0.5, err

        try:
            output = self._llm(
                self._format_prompt(prompt),
                max_tokens=max_tokens,
                temperature=0.1,
                stop=["<|end|>"],
            )
            raw = output.get("choices", [{}])[0].get("text", "")
            return _parse_judge_output(raw)
        except Exception as exc:
            return 0.5, f"Local judge error ({type(exc).__name__}): {exc}"

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        return (
            "<|system|>\n"
            "You are a precise AI quality evaluator. "
            "Think carefully, then output ONLY valid JSON on the last line.\n"
            f"<|user|>\n{prompt}\n"
            "<|assistant|>\n"
        )
