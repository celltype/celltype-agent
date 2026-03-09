"""Evo2 DNA embeddings and generation. Hardware: H100 GPU (7B model ~14GB VRAM)."""
import os, time

def run(dna_sequence="", session_id="", **kwargs):
    if not dna_sequence:
        return {"summary": "Error: No DNA sequence provided.", "error": "no_sequence"}
    clean_seq = dna_sequence.strip().upper().replace(" ","").replace("\n","")
    seq_len = len(clean_seq)
    t0 = time.time()

    try:
        import torch
        from evo2 import Evo2
    except ImportError as e:
        return {"summary": f"Error: evo2 not installed: {e}", "error": "missing_dep"}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vram_before = torch.cuda.memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0

    t_load = time.time()
    model = Evo2("evo2_7b")
    t_load = time.time() - t_load
    vram_after_load = torch.cuda.memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0

    t_infer = time.time()
    input_ids = torch.tensor(model.tokenizer.tokenize(clean_seq), dtype=torch.int).unsqueeze(0).to(device)
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    if isinstance(logits, tuple): logits = logits[0]
    logits_shape = list(logits.shape)
    t_infer = time.time() - t_infer

    vram_peak = torch.cuda.max_memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0

    # Generate DNA extension
    generated_seq = ""
    t_gen = time.time()
    if seq_len < 100:
        try:
            gen = model.generate(prompt_seqs=[clean_seq], n_tokens=50, temperature=1.0, top_k=4)
            generated_seq = gen.sequences[0] if gen.sequences else ""
        except: pass
    t_gen = time.time() - t_gen

    result = {
        "summary": f"Evo2-7B: processed {seq_len}-base DNA sequence. Logits shape: {logits_shape}.",
        "sequence_length": seq_len,
        "logits_shape": logits_shape,
        "embedding_dim": logits_shape[-1] if logits_shape else 0,
        "metrics": {
            "vram_before_model_load_mb": vram_before,
            "vram_after_model_load_mb": vram_after_load,
            "vram_peak_mb": vram_peak,
            "time_model_load_s": round(t_load, 2),
            "time_inference_s": round(t_infer, 2),
            "time_generation_s": round(t_gen, 2),
            "time_total_s": round(time.time()-t0, 2),
        },
    }
    if generated_seq:
        result["generated_extension"] = generated_seq[:200]
        result["summary"] += f" Generated {len(generated_seq)} additional bases."
    return result
