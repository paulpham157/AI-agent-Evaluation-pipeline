# Plan — Local Judge: Qwen3-8B (GGUF Q4_K_M, ~5GB)

## Tổng quan

Thêm khả năng dùng **LLM judge chạy local** (via llama.cpp / llama-cpp-python) làm backend cho evaluation mode "LLM Judge", song song với HF Inference API hiện tại.

Model mục tiêu: **Qwen3-8B-Instruct** (GGUF Q4_K_M, ~5GB) — thế hệ mới nhất của Qwen, hỗ trợ `/think` toggle, instruction-following xuất sắc, đủ nhẹ chạy trên ZeroGPU A100 shared.

---

## Files cần thay đổi

| File | Thay đổi |
|------|----------|
| `src/llm_judge.py` | Thêm `LocalQwenJudge` class (llama.cpp), giữ nguyên `LLMJudge` (Inference API) |
| `app.py` | UI: chọn judge backend (Inference API / Local), nhập model path, pass đúng judge vào `EvalRunner` |
| `.env.example` | Thêm `LOCAL_JUDGE_MODEL_PATH`, `LOCAL_JUDGE_N_CTX` |
| `AGENTS.md` | Ghi lại pattern optional import cho local judge |

---

## Kiến trúc

```
User clicks "LLM Judge" mode in Gradio
  ├── Backend: "Inference API"
  │     → LLMJudge(model_id="Qwen/Qwen3.6-27B", api_key="hf_...")
  │     → calls HF Inference API
  │
  └── Backend: "Local (llama.cpp)"
        → LocalQwenJudge(model_path="/path/to/model.gguf")
        → calls llama-cpp-python Llama()
```

### Interface contract (không đổi)

```python
def score(self, prompt: str) -> Tuple[float, str]:
    """Returns (normalized_score [0.0-1.0], explanation)"""
```

Cả `LLMJudge` và `LocalQwenJudge` đều implement interface này. `EvalRunner` và các evaluator không cần thay đổi — chúng chỉ gọi `judge.score(prompt)`.

---

## Implementation

### 1. `src/llm_judge.py` — thêm `LocalQwenJudge`

```python
class LocalQwenJudge:
    """
    Local judge via llama-cpp-python + GGUF model.
    Optional dependency — guard import with try/except.
    """
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        # lazy init Llama(model_path=..., n_ctx=..., n_gpu_layers=..., verbose=False)

    @property
    def available(self) -> bool:
        return self._llm is not None

    def score(self, prompt: str, max_tokens: int = 512) -> Tuple[float, str]:
        # call self._llm(prompt, max_tokens=..., temperature=0.1, stop=...)
        # parse JSON from output, same _parse() logic as LLMJudge
```

### 2. `app.py` — UI changes

- Radio: **"LLM Judge Backend"** → `Inference API` / `Local (llama.cpp)`
- Khi chọn **Local**: hiện textbox `GGUF model path` + checkbox `Use all GPU layers`
- Khi chọn **Inference API**: hiện textbox `HF Token` (như cũ)
- `run_evaluation()`: khởi tạo judge đúng loại dựa trên backend

### 3. `.env.example`

```ini
# ─── LLM Judge (src/llm_judge.py) ──────────────────────────────────
LLM_JUDGE_BACKEND=inference         # "inference" | "local"
LLM_JUDGE_MODEL=Qwen/Qwen3.6-27B   # Inference API model ID
LOCAL_JUDGE_MODEL_PATH=             # path to GGUF file (for local backend)
LOCAL_JUDGE_N_CTX=8192              # context length for local judge
LOCAL_JUDGE_N_GPU_LAYERS=-1         # -1 = all layers on GPU
```

---

## Yêu cầu

- `llama-cpp-python>=0.3.0` (đã có trong `requirements.txt`)
- GGUF file: Qwen2.5-7B-Instruct Q4_K_M (~4.5GB download từ HF)

## HF model

```
https://huggingface.co/Qwen/Qwen3-8B-GGUF
→ File: Qwen3-8B-Q4_K_M.gguf (5.03 GB)
```

Có thể auto-download bằng `hf_hub_download(repo_id="Qwen/Qwen3-8B-GGUF", filename="Qwen3-8B-Q4_K_M.gguf")`.

> **Lưu ý:** Qwen3-8B dùng chat template `<|im_start|>` / `<|im_end|>`, khác với Qwen2.5 (`<|system|>`). Template đã được cập nhật trong `LocalQwenJudge._format_prompt()`.

---

## Edge cases

- **Import fail**: `llama-cpp-python` chưa install → fallback message, không crash
- **Model file not found**: show error rõ ràng, fallback về heuristic
- **OOM trên ZeroGPU**: user giảm `n_ctx` hoặc dùng model nhỏ hơn (Qwen2.5-1.5B)
- **Inference API + Local cùng lúc**: không support — chọn 1 backend

---

## Testing

1. `python3 -c "from src.llm_judge import LocalQwenJudge; print('OK')"` — import test
2. Start Gradio: `python3 app.py` → tab Configure → chọn "LLM Judge (Local)" → chạy eval
3. CLI test: `python3 scripts/eval_judge.py --judge local --model /path/to/model.gguf` (nếu cần script riêng)
