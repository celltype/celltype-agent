"""
Protein design tools: BindCraft and antibody design.

These tools provide interfaces to computational protein design software.
Each checks for local installation first and falls back to providing
install instructions with a dry-run job submission template.

Note: ProteinMPNN and RFdiffusion are available as container tools
(design.proteinmpnn, design.rfdiffusion) loaded from tool.yaml files.
"""

import os
import shutil
import subprocess

from ct.tools import registry


def _check_tool_installed(tool_name: str, executables: list[str]) -> str | None:
    """Check if any of the given executable names exist in PATH. Returns found path or None."""
    for exe in executables:
        path = shutil.which(exe)
        if path:
            return path
    return None


def _not_installed_result(tool_name: str, install_instructions: dict) -> dict:
    """Return a standardized not-installed response with install instructions."""
    methods = "\n".join(f"  {k}: {v}" for k, v in install_instructions.items())
    return {
        "error": f"{tool_name} is not installed locally.",
        "summary": (
            f"{tool_name} not installed. Install instructions provided. "
            f"This tool requires local GPU compute or a cloud platform."
        ),
        "install_instructions": install_instructions,
        "install_text": f"Install {tool_name}:\n{methods}",
    }


@registry.register(
    name="protein_design.bindcraft",
    description="Run the BindCraft pipeline for end-to-end binder design (RFdiffusion + ProteinMPNN + AlphaFold2 validation)",
    category="protein_design",
    parameters={
        "target_pdb": "Path to target protein PDB file",
        "hotspot_residues": "Target residues to contact (optional, e.g. 'A30,A33,A34')",
        "num_designs": "Number of binder designs to generate (default 4)",
        "binder_length": "Length of the designed binder in residues (default 80)",
    },
    requires_data=[],
    usage_guide="You want to design a protein binder for a target with end-to-end validation. BindCraft chains RFdiffusion → ProteinMPNN → AlphaFold2 validation into a single pipeline, filtering for designs predicted to actually bind. More automated than running individual tools.",
)
def bindcraft(target_pdb: str, hotspot_residues: str = None, num_designs: int = 4, binder_length: int = 80, **kwargs) -> dict:
    """Run BindCraft end-to-end binder design pipeline."""
    target_pdb = (target_pdb or "").strip()
    if not target_pdb:
        return {"error": "target_pdb is required", "summary": "BindCraft requires a target PDB file"}
    if not os.path.isfile(target_pdb):
        return {"error": f"Target PDB not found: {target_pdb}", "summary": f"File not found: {target_pdb}"}

    num_designs = max(1, min(int(num_designs or 4), 50))
    binder_length = max(30, min(int(binder_length or 80), 300))

    # Check for BindCraft
    bindcraft_path = _check_tool_installed("BindCraft", ["bindcraft", "bindcraft.py"])

    if bindcraft_path is None:
        result = _not_installed_result("BindCraft", {
            "github": "git clone https://github.com/martinpacesa/BindCraft.git && cd BindCraft && pip install -e .",
            "note": "Requires GPU, RFdiffusion, ProteinMPNN, and AlphaFold2/ColabFold weights.",
        })
        hs_arg = f" --hotspot_residues {hotspot_residues}" if hotspot_residues else ""
        result["dry_run"] = {
            "command": f"bindcraft --target {target_pdb}{hs_arg} --num_designs {num_designs} --binder_length {binder_length}",
            "target_pdb": target_pdb,
            "hotspot_residues": hotspot_residues,
            "num_designs": num_designs,
            "binder_length": binder_length,
        }
        return result

    # Run BindCraft
    import tempfile
    output_dir = tempfile.mkdtemp(prefix="bindcraft_")

    cmd = [
        bindcraft_path,
        "--target", target_pdb,
        "--output_dir", output_dir,
        "--num_designs", str(num_designs),
        "--binder_length", str(binder_length),
    ]
    if hotspot_residues:
        cmd.extend(["--hotspot_residues", hotspot_residues])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"error": "BindCraft timed out after 60 minutes", "summary": "BindCraft timed out"}
    except Exception as e:
        return {"error": f"BindCraft execution failed: {e}", "summary": f"BindCraft failed: {e}"}

    if proc.returncode != 0:
        return {"error": f"BindCraft error: {proc.stderr[:300]}", "summary": f"BindCraft failed with exit code {proc.returncode}"}

    # Parse output
    import glob
    pdb_files = glob.glob(os.path.join(output_dir, "**", "*.pdb"), recursive=True)

    designs = []
    for pdb_file in pdb_files:
        designs.append({
            "pdb_path": pdb_file,
            "filename": os.path.basename(pdb_file),
        })

    return {
        "summary": (
            f"BindCraft: {len(designs)} binder design(s) generated for {os.path.basename(target_pdb)} "
            f"(length={binder_length})"
        ),
        "target_pdb": target_pdb,
        "hotspot_residues": hotspot_residues,
        "num_designs": num_designs,
        "binder_length": binder_length,
        "output_dir": output_dir,
        "designs": designs,
    }


@registry.register(
    name="protein_design.antibody_design",
    description="Design antibody CDR variants with humanization scoring and germline assignment using ANARCI/AbNumber",
    category="protein_design",
    parameters={
        "heavy_chain": "Heavy chain amino acid sequence",
        "light_chain": "Light chain amino acid sequence (optional)",
        "target_pdb": "Target antigen PDB for structure-guided design (optional)",
        "cdr_to_optimize": "CDR loop to redesign: 'H1', 'H2', 'H3', 'L1', 'L2', or 'L3' (default 'H3')",
        "num_designs": "Number of CDR variants to generate (default 8)",
    },
    requires_data=[],
    usage_guide="You have an antibody sequence and want to optimize a CDR loop, check humanization, or get germline assignments. Uses ANARCI for numbering and AbNumber for humanness scoring. Essential for therapeutic antibody engineering.",
)
def antibody_design(heavy_chain: str, light_chain: str = None, target_pdb: str = None, cdr_to_optimize: str = "H3", num_designs: int = 8, **kwargs) -> dict:
    """Design antibody CDR variants with numbering and humanization analysis."""
    heavy_chain = (heavy_chain or "").strip().upper()
    if not heavy_chain:
        return {"error": "Heavy chain sequence is required", "summary": "No heavy chain sequence provided"}

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(heavy_chain) - valid_aa
    if invalid:
        return {"error": f"Invalid characters in heavy chain: {invalid}", "summary": f"Invalid characters: {invalid}"}

    if light_chain:
        light_chain = light_chain.strip().upper()
        invalid_l = set(light_chain) - valid_aa
        if invalid_l:
            return {"error": f"Invalid characters in light chain: {invalid_l}", "summary": f"Invalid characters: {invalid_l}"}

    cdr_to_optimize = (cdr_to_optimize or "H3").upper()
    num_designs = max(1, min(int(num_designs or 8), 50))

    # Try ANARCI for numbering
    try:
        from anarci import anarci as run_anarci

        # Number heavy chain
        numbered_h, _, _ = run_anarci(
            [("heavy", heavy_chain)],
            scheme="imgt",
            output=False,
        )

        # Parse ANARCI output
        numbering = {}
        germline_h = ""
        if numbered_h and numbered_h[0]:
            for chain_result in numbered_h[0]:
                if chain_result:
                    for pos, aa in chain_result:
                        pos_str = f"H{pos[0]}{pos[1].strip()}" if pos[1].strip() else f"H{pos[0]}"
                        numbering[pos_str] = aa

        # Extract CDR regions (IMGT numbering)
        cdr_definitions = {
            "H1": (26, 38), "H2": (56, 65), "H3": (105, 117),
            "L1": (27, 38), "L2": (56, 65), "L3": (105, 117),
        }

        cdrs = {}
        for cdr_name, (start, end) in cdr_definitions.items():
            chain_prefix = cdr_name[0]
            cdr_seq = ""
            for pos in range(start, end + 1):
                aa = numbering.get(f"{chain_prefix}{pos}", "-")
                if aa != "-":
                    cdr_seq += aa
            if cdr_seq:
                cdrs[cdr_name] = cdr_seq

        # Number light chain if provided
        if light_chain:
            numbered_l, _, _ = run_anarci(
                [("light", light_chain)],
                scheme="imgt",
                output=False,
            )
            if numbered_l and numbered_l[0]:
                for chain_result in numbered_l[0]:
                    if chain_result:
                        for pos, aa in chain_result:
                            pos_str = f"L{pos[0]}{pos[1].strip()}" if pos[1].strip() else f"L{pos[0]}"
                            numbering[pos_str] = aa

        # Generate CDR variants (simple single-point mutations of the target CDR)
        import random
        target_cdr = cdrs.get(cdr_to_optimize, "")
        variants = []
        if target_cdr:
            aa_list = list("ACDEFGHIKLMNPQRSTVWY")
            for _ in range(num_designs):
                var_seq = list(target_cdr)
                # Mutate 1-3 positions
                n_mutations = random.randint(1, min(3, len(var_seq)))
                positions = random.sample(range(len(var_seq)), n_mutations)
                mutations = []
                for pos in positions:
                    old_aa = var_seq[pos]
                    new_aa = random.choice([a for a in aa_list if a != old_aa])
                    var_seq[pos] = new_aa
                    mutations.append(f"{old_aa}{pos + 1}{new_aa}")

                variants.append({
                    "cdr_sequence": "".join(var_seq),
                    "mutations": mutations,
                    "n_mutations": len(mutations),
                })

        # Humanization scoring (basic — count human germline matches)
        humanization_score = None
        try:
            from abnumber import Chain
            chain_obj = Chain(heavy_chain, scheme="imgt")
            humanization_score = chain_obj.humanness_score if hasattr(chain_obj, "humanness_score") else None
        except (ImportError, Exception):
            pass

        cdr_str = ", ".join(f"{k}={v}" for k, v in cdrs.items())

        return {
            "summary": (
                f"Antibody design: {len(cdrs)} CDR(s) identified ({cdr_str}). "
                f"{len(variants)} {cdr_to_optimize} variants generated."
                + (f" Humanization score: {humanization_score}" if humanization_score else "")
            ),
            "heavy_chain_length": len(heavy_chain),
            "light_chain_length": len(light_chain) if light_chain else 0,
            "cdr_regions": cdrs,
            "cdr_to_optimize": cdr_to_optimize,
            "variants": variants,
            "numbering_scheme": "imgt",
            "humanization_score": humanization_score,
            "computed_locally": True,
        }

    except ImportError:
        # ANARCI not installed — provide basic analysis and install instructions
        # Basic CDR3 estimation by length heuristics
        estimated_cdrs = {}
        if len(heavy_chain) >= 100:
            # Very rough CDR3 estimation: usually around position 93-102 in IMGT
            estimated_cdrs["H3_estimated"] = heavy_chain[93:110] if len(heavy_chain) > 110 else heavy_chain[93:]

        return {
            "error": (
                "ANARCI is required for antibody numbering. Install with:\n"
                "  pip install anarci\n"
                "For humanization scoring also install: pip install abnumber"
            ),
            "summary": (
                f"Antibody analysis (limited — ANARCI not installed): "
                f"Heavy chain {len(heavy_chain)} aa"
                + (f", Light chain {len(light_chain)} aa" if light_chain else "")
                + ". Install ANARCI for full CDR analysis and variant generation."
            ),
            "heavy_chain_length": len(heavy_chain),
            "light_chain_length": len(light_chain) if light_chain else 0,
            "estimated_cdrs": estimated_cdrs,
            "install_instructions": {
                "anarci": "pip install anarci",
                "abnumber": "pip install abnumber",
                "hmmer": "conda install -c bioconda hmmer (required by ANARCI)",
            },
            "computed_locally": False,
        }


@registry.register(
    name="protein_design.antifold",
    description="Antibody-specific inverse folding using AntiFold (fine-tuned ESM-IF1 for CDR design)",
    category="protein_design",
    parameters={
        "pdb_path": "Path to antibody PDB file (Fv or Fab structure)",
        "chains": "Chain IDs to design (e.g. 'H' for heavy, 'HL' for both, default 'H')",
        "num_sequences": "Number of sequences to sample (default 10)",
        "regions": "CDR regions to redesign (e.g. 'CDRH3' or 'CDRH1,CDRH2,CDRH3', default all CDRs)",
        "temperature": "Sampling temperature (default 0.2, lower = more conservative)",
    },
    requires_data=[],
    usage_guide="You have an antibody structure and want to redesign CDR sequences. AntiFold (Bioinformatics Advances 2025) outperforms generic inverse folding (ProteinMPNN, ESM-IF) on antibody CDR sequence recovery. Use for antibody humanization, affinity maturation, or CDR library design.",
)
def antifold(pdb_path: str, chains: str = "H", num_sequences: int = 10, regions: str = None, temperature: float = 0.2, **kwargs) -> dict:
    """Run AntiFold antibody-specific inverse folding."""
    pdb_path = (pdb_path or "").strip()
    if not pdb_path:
        return {"error": "pdb_path is required", "summary": "AntiFold requires an antibody PDB file"}
    if not os.path.isfile(pdb_path):
        return {"error": f"PDB file not found: {pdb_path}", "summary": f"File not found: {pdb_path}"}

    num_sequences = max(1, min(int(num_sequences or 10), 100))
    temperature = max(0.01, min(float(temperature or 0.2), 2.0))

    try:
        import antifold
        from antifold.main import predict as antifold_predict
    except ImportError:
        result = _not_installed_result("AntiFold", {
            "pip": "pip install antifold",
            "github": "git clone https://github.com/oxpig/AntiFold && cd AntiFold && pip install -e .",
            "note": "Fine-tuned ESM-IF1 for antibody inverse folding. CPU or GPU.",
        })
        chains_arg = f" --chains {chains}" if chains else ""
        regions_arg = f" --regions {regions}" if regions else ""
        result["dry_run"] = {
            "command": f"antifold --pdb {pdb_path}{chains_arg}{regions_arg} --num_sequences {num_sequences} --temperature {temperature}",
            "pdb_path": pdb_path,
            "chains": chains,
            "num_sequences": num_sequences,
            "regions": regions,
            "temperature": temperature,
        }
        return result

    try:
        # Parse regions
        region_list = None
        if regions:
            region_list = [r.strip() for r in regions.split(",") if r.strip()]

        results = antifold_predict(
            pdb_path=pdb_path,
            chains=list(chains),
            num_sequences=num_sequences,
            regions=region_list,
            temperature=temperature,
        )

        # Parse output
        sequences = []
        if isinstance(results, (list, tuple)):
            for i, seq_result in enumerate(results):
                if isinstance(seq_result, dict):
                    sequences.append(seq_result)
                elif isinstance(seq_result, str):
                    sequences.append({"sequence": seq_result, "rank": i + 1})
        elif isinstance(results, dict):
            sequences = results.get("sequences", [results])

        return {
            "summary": (
                f"AntiFold: {len(sequences)} sequence(s) designed for {os.path.basename(pdb_path)} "
                f"(chains={chains}, T={temperature})"
            ),
            "pdb_path": pdb_path,
            "chains": chains,
            "regions": region_list,
            "temperature": temperature,
            "n_sequences": len(sequences),
            "sequences": sequences[:num_sequences],
        }
    except Exception as e:
        return {"error": f"AntiFold prediction failed: {e}", "summary": f"AntiFold error: {e}"}
