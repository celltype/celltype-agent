"""Cell2Sentence Gemma 27B local inference entrypoint."""

import os
import re
import time
from pathlib import Path
from typing import Tuple


MODEL_ID = os.environ.get("CELL2SENTENCE_MODEL_ID", "vandijklab/C2S-Scale-Gemma-2-27B")
CACHE_DIR = os.environ.get("CELL2SENTENCE_CACHE", "/root/.cache/cell2sentence")

_MODEL = None
_TOKENIZER = None


def _normalize_sentence(cell_sentence: str) -> str:
    return " ".join((cell_sentence or "").strip().split())


def _build_prompt(cell_sentence: str, organism: str, num_genes: int, task_prompt: str) -> Tuple[str, str]:
    organism = (organism or "Homo sapiens").strip() or "Homo sapiens"
    if task_prompt and task_prompt.strip():
        prompt = (
            f"The following is a list of {num_genes} gene names ordered by descending expression level "
            f"in a {organism} cell.\n"
            f"Cell sentence: {cell_sentence}.\n"
            f"Task: {task_prompt.strip()}\n"
            "Response:"
        )
        response_prefix = "Response:"
        return prompt, response_prefix

    prompt = (
        f"The following is a list of {num_genes} gene names ordered by descending expression level in a {organism} cell. "
        "Your task is to give the cell type which this cell belongs to based on its gene expression.\n"
        f"Cell sentence: {cell_sentence}.\n"
        "The cell type corresponding to these genes is:"
    )
    return prompt, "The cell type corresponding to these genes is:"


def _sanitize_response(text: str) -> str:
    cleaned = re.sub(r"<ctrl\d+>", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _cache_has_model(cache_dir: str) -> bool:
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False
    patterns = (
        "*model*.safetensors",
        "*pytorch_model*.bin",
        "*tokenizer*.json",
        "*tokenizer.model",
    )
    for pattern in patterns:
        if any(cache_path.rglob(pattern)):
            return True
    return False


def _load_model():
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    model_kwargs = {
        "cache_dir": CACHE_DIR,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    _MODEL.eval()
    return _MODEL, _TOKENIZER


def run(
    cell_sentence: str = "",
    organism: str = "Homo sapiens",
    num_genes: int = 1000,
    task_prompt: str = "",
    max_new_tokens: int = 32,
    temperature: float = 0,
    top_p: float = 0.95,
    session_id: str = "",
    **kwargs,
) -> dict:
    clean_sentence = _normalize_sentence(cell_sentence)
    if not clean_sentence:
        return {"summary": "Error: No cell_sentence provided.", "error": "no_cell_sentence"}

    try:
        num_genes = max(1, int(num_genes))
    except (TypeError, ValueError):
        num_genes = len(clean_sentence.split())

    try:
        max_new_tokens = min(512, max(1, int(max_new_tokens)))
    except (TypeError, ValueError):
        max_new_tokens = 32

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.0

    try:
        top_p = float(top_p)
    except (TypeError, ValueError):
        top_p = 0.95

    t0 = time.time()

    try:
        import torch
    except ImportError as exc:
        return {"summary": f"Error: torch not installed: {exc}", "error": "missing_dep"}

    if not torch.cuda.is_available():
        return {
            "summary": "Error: Cell2Sentence Gemma 27B requires a CUDA GPU for local execution.",
            "error": "no_gpu",
        }

    weights_cache_hit = _cache_has_model(CACHE_DIR)
    vram_before = torch.cuda.memory_allocated() // (1024 * 1024)
    t_load_start = time.time()
    try:
        model, tokenizer = _load_model()
    except Exception as exc:
        return {
            "summary": f"Error: Failed to load {MODEL_ID}: {exc}",
            "error": "model_load_failed",
        }
    t_load = time.time() - t_load_start
    vram_after_load = torch.cuda.memory_allocated() // (1024 * 1024)

    prompt, response_prefix = _build_prompt(clean_sentence, organism, num_genes, task_prompt)

    t_infer_start = time.time()
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": max(0.0, min(1.0, top_p)),
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
    except Exception as exc:
        return {
            "summary": f"Error: Generation failed for {MODEL_ID}: {exc}",
            "error": "generation_failed",
        }
    t_infer = time.time() - t_infer_start

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not response_text:
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    response_text = _sanitize_response(response_text)

    vram_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
    prediction = response_text.splitlines()[0].strip() if response_text else ""

    return {
        "summary": (
            f"Cell2Sentence Gemma 27B generated a response for a {len(clean_sentence.split())}-gene cell sentence. "
            f"Preview: {prediction[:160]}"
        ),
        "model_id": MODEL_ID,
        "cache_dir": CACHE_DIR,
        "prompt": prompt,
        "response_prefix": response_prefix,
        "response": response_text,
        "prediction": prediction,
        "input_gene_count": len(clean_sentence.split()),
        "task_prompt": task_prompt or "default_cell_type_prediction",
        "metrics": {
            "weights_cache_hit": weights_cache_hit,
            "weights_cache_path": CACHE_DIR,
            "vram_before_model_load_mb": vram_before,
            "vram_after_model_load_mb": vram_after_load,
            "vram_peak_mb": vram_peak,
            "time_model_load_s": round(t_load, 2),
            "time_inference_s": round(t_infer, 2),
            "time_total_s": round(time.time() - t0, 2),
        },
    }
