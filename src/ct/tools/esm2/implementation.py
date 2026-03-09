"""ESM2-650M protein embeddings. Hardware: A10G GPU, 16GB VRAM."""
import time

def run(sequence: str = "", session_id: str = "", **kwargs) -> dict:
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    clean_seq = sequence.strip().upper().replace(" ", "").replace("\n", "")
    seq_len = len(clean_seq)
    t0 = time.time()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except (ImportError, Exception):
        return {"summary": f"ESM2: {seq_len}-residue (no GPU libs).", "error": "missing_dep"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_before = torch.cuda.memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0

    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    model.eval()
    t_load = time.time() - t_load_start
    vram_after_load = torch.cuda.memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0

    t_infer_start = time.time()
    with torch.no_grad():
        inputs = tokenizer(clean_seq, return_tensors="pt", add_special_tokens=True).to(device)
        outputs = model(**inputs)
    t_infer = time.time() - t_infer_start
    vram_peak = torch.cuda.max_memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0

    last_hidden = outputs.last_hidden_state[0]
    per_residue = last_hidden[1:-1]
    pooled = per_residue.mean(dim=0)
    embedding_shape = list(per_residue.shape)
    pooled_shape = list(pooled.shape)

    return {
        "summary": f"ESM2-650M: generated embeddings for {seq_len}-residue protein. Per-residue shape: ({embedding_shape[0]}, {embedding_shape[1]}), pooled: ({pooled_shape[0]},).",
        "embedding_shape": embedding_shape,
        "pooled_embedding_shape": pooled_shape,
        "pooled_embedding": pooled.cpu().tolist()[:10],
        "num_residues": seq_len,
        "metrics": {
            "vram_before_model_load_mb": vram_before,
            "vram_after_model_load_mb": vram_after_load,
            "vram_peak_mb": vram_peak,
            "time_model_load_s": round(t_load, 2),
            "time_inference_s": round(t_infer, 2),
            "time_total_s": round(time.time() - t0, 2),
        },
    }
