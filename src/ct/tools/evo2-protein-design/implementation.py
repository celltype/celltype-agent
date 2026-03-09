"""Evo2 protein design. Hardware: H100 GPU."""
import os, time

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def translate_dna(dna):
    protein = []
    for i in range(0, len(dna)-2, 3):
        aa = CODON_TABLE.get(dna[i:i+3], 'X')
        if aa == '*': break
        protein.append(aa)
    return ''.join(protein)

def run(target_function="", constraints="", session_id="", **kwargs):
    if not target_function:
        return {"summary": "Error: No target function.", "error": "no_target"}
    t0 = time.time()
    try:
        import torch; from evo2 import Evo2
    except ImportError as e:
        return {"summary": f"Error: {e}", "error": "missing_dep"}
    
    vram_before = torch.cuda.memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0
    t_load = time.time()
    model = Evo2("evo2_7b")
    t_load = time.time() - t_load
    vram_after_load = torch.cuda.memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0
    
    t_gen = time.time()
    dna_seqs, protein_seqs = [], []
    for i in range(3):
        try:
            gen = model.generate(prompt_seqs=["ATG"], n_tokens=300, temperature=0.8+i*0.1, top_k=4)
            if gen.sequences:
                dna = gen.sequences[0]
                protein = translate_dna(dna)
                if len(protein) >= 10:
                    dna_seqs.append(dna[:len(protein)*3+3])
                    protein_seqs.append(protein)
        except: pass
    t_gen = time.time() - t_gen
    vram_peak = torch.cuda.max_memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0
    
    if not protein_seqs:
        return {"summary": "No valid protein-coding sequences found.", "error": "no_valid_proteins"}
    
    return {
        "summary": f"Evo2 Protein Design: {len(protein_seqs)} candidates. Lengths: {[len(p) for p in protein_seqs]}.",
        "dna_sequences": dna_seqs, "protein_sequences": protein_seqs, "num_designs": len(protein_seqs),
        "metrics": {
            "vram_before_model_load_mb": vram_before, "vram_after_model_load_mb": vram_after_load,
            "vram_peak_mb": vram_peak, "time_model_load_s": round(t_load,2),
            "time_generation_s": round(t_gen,2), "time_total_s": round(time.time()-t0,2),
        },
    }
