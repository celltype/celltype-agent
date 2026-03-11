"""Evo2 DNA generation and analysis. Hardware: H100 GPU (7B model ~14GB VRAM)."""
import os
import time


def run(dna_sequence="", num_tokens=100, temperature=0.7, top_k=3, top_p=0,
        random_seed=None, session_id="", **kwargs):
    if not dna_sequence:
        return {"summary": "Error: No DNA sequence provided.", "error": "no_sequence"}

    clean_seq = dna_sequence.strip().upper().replace(" ", "").replace("\n", "")
    seq_len = len(clean_seq)

    # Validate
    valid_bases = set("ACGT")
    invalid = set(clean_seq) - valid_bases
    if invalid:
        return {
            "summary": f"Error: Invalid characters in DNA sequence: {invalid}",
            "error": "invalid_sequence",
        }

    # Coerce parameters
    try:
        num_tokens = int(num_tokens) if num_tokens is not None else 100
    except (TypeError, ValueError):
        num_tokens = 100
    try:
        temperature = float(temperature) if temperature is not None else 0.7
    except (TypeError, ValueError):
        temperature = 0.7
    try:
        top_k = int(top_k) if top_k is not None else 3
    except (TypeError, ValueError):
        top_k = 3
    try:
        top_p = float(top_p) if top_p is not None else 0
    except (TypeError, ValueError):
        top_p = 0

    t0 = time.time()

    try:
        import torch
        from evo2 import Evo2
    except ImportError as e:
        return {"summary": f"Error: evo2 not installed: {e}", "error": "missing_dep"}

    if not torch.cuda.is_available():
        return {
            "summary": "Error: Evo2 requires a GPU (H100) but none is available. This is a transient infrastructure issue — please retry.",
            "error": "no_gpu",
        }
    device = "cuda:0"
    vram_before = torch.cuda.memory_allocated() // (1024 * 1024)

    if random_seed is not None:
        try:
            torch.manual_seed(int(random_seed))
        except (TypeError, ValueError):
            pass

    t_load = time.time()
    model = Evo2("evo2_7b")
    t_load = time.time() - t_load
    vram_after_load = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    # Generate DNA — always, matching NVIDIA behavior
    t_gen = time.time()
    generated_sequence = ""
    sampled_probs = []
    try:
        gen_kwargs = {
            "prompt_seqs": [clean_seq],
            "n_tokens": num_tokens,
            "temperature": max(temperature, 1e-6),
            "top_k": top_k,
        }
        if top_p > 0:
            gen_kwargs["top_p"] = top_p

        gen = model.generate(**gen_kwargs)
        generated_sequence = gen.sequences[0] if gen.sequences else ""

        # gen.logits is a list of tensors, shape (batch, num_tokens, vocab_size)
        # gen.logprobs_mean is the mean log-probability per sequence
        # Compute per-token sampled probabilities from logits
        if gen.logits and len(gen.logits) > 0:
            try:
                logits_tensor = gen.logits[0]  # first batch group
                # Get probabilities via softmax
                probs_tensor = torch.nn.functional.softmax(logits_tensor.float(), dim=-1)
                # Tokenize the generated sequence to get the token IDs that were sampled
                gen_tokens = model.tokenizer.tokenize(generated_sequence)
                if isinstance(gen_tokens, list):
                    gen_tokens = torch.tensor(gen_tokens, dtype=torch.long)
                # For each generated token, get the probability of the token that was sampled
                # logits shape: (batch, seq_len, vocab). We want probs for batch=0
                for i in range(min(len(gen_tokens), probs_tensor.shape[1])):
                    tid = gen_tokens[i].item() if hasattr(gen_tokens[i], 'item') else int(gen_tokens[i])
                    if tid < probs_tensor.shape[-1]:
                        sampled_probs.append(round(float(probs_tensor[0, i, tid].item()), 6))
            except Exception as e:
                # Log but don't fail — probs are optional
                import sys
                print(f"Warning: could not extract sampled_probs: {e}", file=sys.stderr)

    except Exception as e:
        return {
            "summary": f"Error: Evo2 generation failed: {e}",
            "error": str(e),
        }
    t_gen = time.time() - t_gen

    vram_peak = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    generated_part = generated_sequence if generated_sequence else ""
    full_sequence = clean_seq + generated_part

    def _sanitize(obj):
        """Ensure all values are JSON-serializable (no numpy/torch types)."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    result = {
        "summary": (
            f"Evo2: generated {len(generated_part)} bases from {seq_len}-base input. "
            f"Total output: {len(full_sequence)} bases."
        ),
        "sequence": full_sequence,
        "generated_sequence": generated_part,
        "input_length": seq_len,
        "output_length": len(full_sequence),
        "num_generated_tokens": len(generated_part),
        "metrics": {
            "vram_before_model_load_mb": vram_before,
            "vram_after_model_load_mb": vram_after_load,
            "vram_peak_mb": vram_peak,
            "time_model_load_s": round(t_load, 2),
            "time_generation_s": round(t_gen, 2),
            "time_total_s": round(time.time() - t0, 2),
        },
    }

    if sampled_probs:
        result["sampled_probs"] = sampled_probs

    # Mean log-probability from the model
    if hasattr(gen, "logprobs_mean") and gen.logprobs_mean:
        result["logprobs_mean"] = round(float(gen.logprobs_mean[0]), 6)

    # Force JSON round-trip to ensure all values are serializable
    import json
    import numpy as np

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if hasattr(o, 'item'): return o.item()
            return super().default(o)

    return json.loads(json.dumps(_sanitize(result), cls=_Encoder))
