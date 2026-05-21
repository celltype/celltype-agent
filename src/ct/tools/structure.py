"""
Structure prediction tools: AlphaFold fetch, docking, MD simulation, FEP, binding sites.

ternary_predict and batch_screen require the TernaryPred sister project
(github.com/celltype/TernaryPred) to be installed locally. All other tools
work standalone.
"""

import subprocess
import sys
import json
from pathlib import Path
from ct.tools import registry
from ct.tools.http_client import request

TERNARYPRED_DIR = Path.home() / "Projects" / "CellType" / "TernaryPred"


@registry.register(
    name="structure.ternary_predict",
    description="Predict ternary complex structure (E3 ligase + compound + target) using DeepTernary (requires TernaryPred installation)",
    category="structure",
    parameters={
        "smiles": "Compound SMILES string",
        "target_pdb": "Path to target protein PDB",
        "e3": "E3 ligase: CRBN or VHL",
    },
    usage_guide="You want to predict how a molecular glue or PROTAC forms a ternary complex between E3 ligase and target. Use when you have a compound SMILES and target structure. Requires TernaryPred installation.",
)
def ternary_predict(smiles: str, target_pdb: str, e3: str = "CRBN",
                    name: str = "prediction", **kwargs) -> dict:
    """Predict ternary complex using DeepTernary via TernaryPred wrapper."""
    script = TERNARYPRED_DIR / "scripts" / "predict_deepternary.py"
    e3_path = TERNARYPRED_DIR / "data" / "e3_structures" / f"{'crbn_5fqd' if e3 == 'CRBN' else 'vhl_4w9h'}.pdb"

    if not script.exists():
        return {
            "error": (
                f"TernaryPred not installed. This tool requires the TernaryPred sister project.\n"
                f"Install: git clone git@github.com:celltype/TernaryPred.git {TERNARYPRED_DIR}"
            ),
            "summary": "Ternary prediction unavailable — TernaryPred not installed",
        }

    cmd = [
        sys.executable, str(script), "single",
        "--smiles", smiles,
        "--e3", str(e3_path),
        "--target", target_pdb,
        "--name", name,
        "--device", "cuda",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        return {"error": f"DeepTernary failed: {result.stderr[:500]}", "summary": f"DeepTernary failed: {result.stderr[:500]}"}
    return {
        "summary": f"Ternary complex predicted for {name}",
        "stdout": result.stdout,
        "e3": e3,
    }


@registry.register(
    name="structure.batch_screen",
    description="Screen compounds against a protein target panel for ternary compatibility (requires TernaryPred installation)",
    category="structure",
    parameters={
        "compounds_csv": "CSV with compound_id and smiles columns",
        "targets_csv": "CSV with target_id and structure_path columns",
        "e3": "E3 ligase: CRBN or VHL",
    },
    usage_guide="You need to screen many compounds against many targets for ternary complex formation. Use for large-scale virtual screening campaigns. Long-running — launches background process.",
)
def batch_screen(compounds_csv: str, targets_csv: str, e3: str = "CRBN",
                 max_compounds: int = None, max_targets: int = None, **kwargs) -> dict:
    """Batch ternary screening via TernaryPred."""
    script = TERNARYPRED_DIR / "scripts" / "predict_deepternary.py"
    e3_path = TERNARYPRED_DIR / "data" / "e3_structures" / f"{'crbn_5fqd' if e3 == 'CRBN' else 'vhl_4w9h'}.pdb"

    if not script.exists():
        return {
            "error": (
                f"TernaryPred not installed. This tool requires the TernaryPred sister project.\n"
                f"Install: git clone git@github.com:celltype/TernaryPred.git {TERNARYPRED_DIR}"
            ),
            "summary": "Batch screening unavailable — TernaryPred not installed",
        }

    cmd = [
        sys.executable, str(script), "batch",
        "--compounds", compounds_csv,
        "--targets", targets_csv,
        "--e3", str(e3_path),
        "--outdir", str(TERNARYPRED_DIR / "predictions" / "ternary_complexes"),
        "--resume",
    ]
    if max_compounds:
        cmd += ["--max-compounds", str(max_compounds)]
    if max_targets:
        cmd += ["--max-targets", str(max_targets)]

    # This is a long-running process — run in background
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return {
        "summary": f"Batch screening started (PID: {result.pid})",
        "pid": result.pid,
        "note": "Long-running process. Check TernaryPred/predictions/ for results.",
    }


@registry.register(
    name="structure.alphafold_fetch",
    description="Download AlphaFold predicted structure for a protein",
    category="structure",
    parameters={"uniprot_id": "UniProt ID"},
    usage_guide="You need a 3D structure for a target protein and no experimental structure is available. Fetches AlphaFold prediction. Use before ternary_predict or structure analysis.",
)
def alphafold_fetch(uniprot_id: str, **kwargs) -> dict:
    """Download AlphaFold structure for a protein."""
    cache_dir = Path.home() / ".ct" / "cache" / "alphafold"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"

    if output_path.exists():
        return {
            "summary": f"AlphaFold structure for {uniprot_id} (cached)",
            "path": str(output_path),
            "cached": True,
        }

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response, error = request(
        "GET",
        url,
        timeout=30,
        retries=2,
        raise_for_status=False,
    )
    if error:
        return {"error": f"Failed to fetch AlphaFold structure: {error}", "summary": f"Failed to fetch AlphaFold structure: {error}"}
    if response.status_code == 200 and response.text.startswith("HEADER"):
        output_path.write_text(response.text)
        return {
            "summary": f"Downloaded AlphaFold structure for {uniprot_id}",
            "path": str(output_path),
            "cached": False,
        }
    else:
        return {"error": f"AlphaFold structure not available for {uniprot_id} (HTTP {response.status_code})", "summary": f"AlphaFold structure not available for {uniprot_id} (HTTP {response.status_code})"}
@registry.register(
    name="structure.compound_3d",
    description="Generate 3D conformer from SMILES and save as SDF",
    category="structure",
    parameters={"smiles": "SMILES string", "output_path": "Output SDF path"},
    usage_guide="You need a 3D structure for a small molecule compound. Use before docking or ternary complex prediction when you only have a SMILES string.",
)
def compound_3d(smiles: str, output_path: str = None, **kwargs) -> dict:
    """Generate 3D conformer for a compound."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Invalid SMILES: {smiles}"}
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    if output_path:
        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()

    return {
        "summary": f"3D conformer generated ({Descriptors.MolWt(mol):.1f} Da, {mol.GetNumAtoms()} atoms)",
        "smiles": smiles,
        "n_atoms": mol.GetNumAtoms(),
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------


def _resolve_pdb(target_pdb: str) -> dict:
    """Resolve a target identifier to a local PDB file path.

    Accepts either a local file path or a UniProt ID.  When a UniProt ID is
    given the AlphaFold structure is fetched first via *alphafold_fetch*.

    Returns ``{"path": str}`` on success or ``{"error": str}`` on failure.
    """
    p = Path(target_pdb)
    if p.exists():
        return {"path": str(p)}

    # Treat as UniProt ID — try AlphaFold download
    result = alphafold_fetch(target_pdb)
    if "error" in result:
        return {"error": f"Could not resolve target '{target_pdb}': {result['error']}", "summary": f"Could not resolve target '{target_pdb}': {result['error']}"}
    return {"path": result["path"]}


def _prepare_ligand_pdbqt(smiles: str, work_dir: Path) -> dict:
    """Generate a 3D conformer from SMILES and convert to PDBQT for Vina.

    Returns ``{"path": str, "mol": rdkit.Mol}`` on success,
    ``{"error": str}`` on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {"error": "RDKit is required for ligand preparation (pip install rdkit)", "summary": "RDKit is required for ligand preparation (pip install rdkit)"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Invalid SMILES: {smiles}"}
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if res != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    sdf_path = work_dir / "ligand.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    pdbqt_path = work_dir / "ligand.pdbqt"

    # Try obabel conversion
    try:
        conv = subprocess.run(
            ["obabel", str(sdf_path), "-O", str(pdbqt_path), "--gen3d"],
            capture_output=True, text=True, timeout=30,
        )
        if conv.returncode == 0 and pdbqt_path.exists():
            return {"path": str(pdbqt_path), "mol": mol, "obabel_fallback": False}
    except FileNotFoundError:
        pass

    # Fallback: write a minimal PDBQT from coordinates
    # Note: lacks proper atom types and charges — adequate for scoring but
    # install Open Babel for production docking: conda install -c conda-forge openbabel
    conf = mol.GetConformer()
    lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        element = atom.GetSymbol()
        lines.append(
            f"ATOM  {i+1:5d}  {element:<3s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    "
            f"{element:>2s}"
        )
    pdbqt_path.write_text("\n".join(lines) + "\n")
    return {"path": str(pdbqt_path), "mol": mol, "obabel_fallback": True}


def _detect_search_box(pdb_path: str) -> dict:
    """Compute a bounding-box centre and size from PDB ATOM coordinates.

    Returns ``{"center_x", "center_y", "center_z", "size_x", "size_y",
    "size_z"}`` with a 10 A padding on each side.
    """
    xs, ys, zs = [], [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except (ValueError, IndexError):
                    continue
    if not xs:
        return {"error": "No atoms found in PDB", "summary": "No atoms found in PDB"}
    padding = 10.0
    return {
        "center_x": round((min(xs) + max(xs)) / 2, 2),
        "center_y": round((min(ys) + max(ys)) / 2, 2),
        "center_z": round((min(zs) + max(zs)) / 2, 2),
        "size_x": round(max(xs) - min(xs) + 2 * padding, 2),
        "size_y": round(max(ys) - min(ys) + 2 * padding, 2),
        "size_z": round(max(zs) - min(zs) + 2 * padding, 2),
    }


@registry.register(
    name="structure.dock",
    description="Molecular docking: dock a ligand (SMILES) into a target protein (PDB path or UniProt ID)",
    category="structure",
    parameters={
        "smiles": "Ligand SMILES string",
        "target_pdb": "Path to target PDB file or UniProt ID for AlphaFold fetch",
        "method": "Docking method: vina (default), diffdock, gnina",
        "n_poses": "Number of docking poses to generate (default 5)",
    },
    usage_guide=(
        "You want to predict how a small molecule binds to a protein target. "
        "Use Vina for fast local docking, DiffDock for GPU-accelerated deep-learning "
        "docking, or gnina for CNN-scored docking. Returns binding poses with "
        "predicted affinities."
    ),
)
def dock(smiles: str, target_pdb: str, method: str = "vina",
         n_poses: int = 5, **kwargs) -> dict:
    """Dock a ligand into a protein target.

    * method='vina'    — runs AutoDock Vina locally if installed, else submits
                         to cloud compute.
    * method='diffdock' — always submitted as cloud GPU job.
    * method='gnina'   — runs gnina locally if installed, else cloud.
    """
    import tempfile

    valid_methods = ("vina", "diffdock", "gnina")
    if method not in valid_methods:
        return {"error": f"Unknown docking method '{method}'. Choose from: {', '.join(valid_methods)}", "summary": f"Unknown docking method '{method}'. Choose from: {', '.join(valid_methods)}"}
    # Resolve target PDB
    target = _resolve_pdb(target_pdb)
    if "error" in target:
        return {"error": target["error"], "summary": f"Docking failed: {target['error']}"}
    pdb_path = target["path"]

    # GPU-only methods go straight to cloud
    if method == "diffdock":
        from ct.tools.compute import submit_job
        job_result = submit_job(
            job_type="molecular_docking",
            params={
                "smiles": smiles,
                "target_pdb": pdb_path,
                "method": "diffdock",
                "n_poses": n_poses,
            },
            dry_run=kwargs.get("dry_run", True),
        )
        if "error" in job_result:
            return {
                "error": job_result["error"],
                "summary": f"DiffDock submission failed: {job_result['error']}",
            }
        return {
            "summary": f"DiffDock docking submitted for {smiles[:40]} into {Path(pdb_path).stem} ({n_poses} poses)",
            "method": "diffdock",
            "job": job_result,
        }

    # Vina / gnina — try local first
    work_dir = Path(tempfile.mkdtemp(prefix="ct_dock_"))

    # Prepare ligand
    lig = _prepare_ligand_pdbqt(smiles, work_dir)
    if "error" in lig:
        return {"error": lig["error"], "summary": f"Ligand preparation failed: {lig['error']}"}

    # Compute search box from target
    box = _detect_search_box(pdb_path)
    if "error" in box:
        return {"error": box["error"], "summary": f"Search box detection failed: {box['error']}"}

    binary = method  # "vina" or "gnina"
    output_path = work_dir / "docking_out.pdbqt"

    # Check if binary is available locally
    try:
        subprocess.run([binary, "--version"], capture_output=True, timeout=5)
        local_available = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        local_available = False

    if not local_available:
        # Submit as cloud job
        from ct.tools.compute import submit_job
        job_result = submit_job(
            job_type="molecular_docking",
            params={
                "smiles": smiles,
                "target_pdb": pdb_path,
                "method": method,
                "n_poses": n_poses,
                "search_box": box,
            },
            dry_run=kwargs.get("dry_run", True),
        )
        if "error" in job_result:
            return {
                "error": job_result["error"],
                "summary": f"{method} not installed locally; cloud submission failed: {job_result['error']}",
            }
        return {
            "summary": (
                f"{method} not installed locally — submitted cloud docking job for "
                f"{smiles[:40]} into {Path(pdb_path).stem}"
            ),
            "method": method,
            "local": False,
            "job": job_result,
        }

    # Run locally
    cmd = [
        binary,
        "--receptor", pdb_path,
        "--ligand", lig["path"],
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--num_modes", str(n_poses),
        "--out", str(output_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"error": f"{method} timed out after 300s", "summary": f"Docking timed out"}

    if proc.returncode != 0:
        return {"error": f"{method} failed: {proc.stderr[:500]}", "summary": f"Docking failed"}

    # Parse output for binding affinities
    poses = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        # Vina output: mode | affinity | rmsd_lb | rmsd_ub
        if len(parts) >= 4:
            try:
                mode = int(parts[0])
                affinity = float(parts[1])
                rmsd_lb = float(parts[2])
                rmsd_ub = float(parts[3])
                poses.append({
                    "mode": mode,
                    "affinity_kcal_mol": affinity,
                    "rmsd_lb": rmsd_lb,
                    "rmsd_ub": rmsd_ub,
                })
            except (ValueError, IndexError):
                continue

    best = poses[0]["affinity_kcal_mol"] if poses else "N/A"

    obabel_note = ""
    if lig.get("obabel_fallback"):
        obabel_note = " (Note: Open Babel not installed — ligand PDBQT may lack proper atom types. Install: conda install -c conda-forge openbabel)"

    return {
        "summary": (
            f"Docked {smiles[:40]} into {Path(pdb_path).stem}: "
            f"best pose {best} kcal/mol ({len(poses)} poses){obabel_note}"
        ),
        "method": method,
        "local": True,
        "poses": poses,
        "output_path": str(output_path),
        "search_box": box,
    }


# ---------------------------------------------------------------------------
# Molecular dynamics simulation
# ---------------------------------------------------------------------------

@registry.register(
    name="structure.md_simulate",
    description="Submit a molecular dynamics simulation job (OpenMM/GROMACS) to cloud GPU compute",
    category="structure",
    parameters={
        "pdb_path": "Path to input PDB structure",
        "duration_ns": "Simulation duration in nanoseconds (default 10.0)",
        "forcefield": "Force field: amber14 (default), charmm36, opls",
        "temperature_k": "Temperature in Kelvin (default 300.0)",
    },
    usage_guide=(
        "You want to run an MD simulation to study protein dynamics, ligand "
        "stability in a binding pocket, or conformational sampling. This is "
        "always a cloud GPU job — too computationally intensive for local execution."
    ),
)
def md_simulate(pdb_path: str, duration_ns: float = 10.0,
                forcefield: str = "amber14", temperature_k: float = 300.0,
                **kwargs) -> dict:
    """Submit an MD simulation to cloud GPU compute.

    Prepares an OpenMM/GROMACS job configuration and submits via
    ``compute.submit_job``.  Always a cloud job.
    """
    valid_forcefields = ("amber14", "charmm36", "opls")
    if forcefield not in valid_forcefields:
        return {"error": f"Unknown forcefield '{forcefield}'. Choose from: {', '.join(valid_forcefields)}", "summary": f"Unknown forcefield '{forcefield}'. Choose from: {', '.join(valid_forcefields)}"}
    if duration_ns <= 0:
        return {"error": "duration_ns must be positive", "summary": "duration_ns must be positive"}
    if temperature_k <= 0:
        return {"error": "temperature_k must be positive", "summary": "temperature_k must be positive"}
    pdb = Path(pdb_path)
    if not pdb.exists():
        return {"error": f"PDB file not found: {pdb_path}",
                "summary": f"MD simulation failed: PDB file not found"}

    # Estimate runtime: ~1 ns/hr on a single A100 for a typical protein
    estimated_hours = duration_ns * 1.0  # rough heuristic
    protein_name = pdb.stem

    config = {
        "pdb_path": str(pdb),
        "duration_ns": duration_ns,
        "forcefield": forcefield,
        "temperature_k": temperature_k,
        "integrator": "LangevinMiddle",
        "timestep_fs": 2.0,
        "solvent": "tip3p",
        "ionic_strength_M": 0.15,
        "reporting_interval_ps": 10.0,
        "platform": "CUDA",
    }

    from ct.tools.compute import submit_job
    job_result = submit_job(
        job_type="molecular_dynamics",
        params={
            "n_samples": 1,
            "config": config,
        },
        dry_run=kwargs.get("dry_run", True),
    )

    if "error" in job_result:
        return {
            "error": job_result["error"],
            "summary": f"MD simulation submission failed: {job_result['error']}",
        }

    return {
        "summary": (
            f"MD simulation submitted: {protein_name} for {duration_ns}ns at "
            f"{temperature_k}K ({forcefield})"
            + (f" (job: {job_result.get('job_id', 'dry-run')})" if not job_result.get("dry_run") else " [DRY RUN]")
        ),
        "config": config,
        "estimated_hours": round(estimated_hours, 1),
        "job": job_result,
    }


# ---------------------------------------------------------------------------
# Free energy perturbation
# ---------------------------------------------------------------------------

@registry.register(
    name="structure.fep",
    description="Submit a free energy perturbation (FEP) calculation for relative binding free energy between two ligands",
    category="structure",
    parameters={
        "smiles_a": "SMILES for ligand A",
        "smiles_b": "SMILES for ligand B",
        "target_pdb": "Path to target protein PDB or UniProt ID",
        "method": "FEP method: openmm (default), gromacs",
    },
    usage_guide=(
        "You want to predict the relative binding free energy difference "
        "between two similar ligands to a target. Use for lead optimization "
        "when you need to rank-order compound modifications by binding affinity. "
        "Always a cloud GPU job."
    ),
)
def fep(smiles_a: str, smiles_b: str, target_pdb: str,
        method: str = "openmm", **kwargs) -> dict:
    """Submit an FEP calculation for relative binding free energy.

    Prepares the ligand pair and target, then submits to cloud compute.
    """
    valid_methods = ("openmm", "gromacs")
    if method not in valid_methods:
        return {"error": f"Unknown FEP method '{method}'. Choose from: {', '.join(valid_methods)}", "summary": f"Unknown FEP method '{method}'. Choose from: {', '.join(valid_methods)}"}
    # Validate SMILES
    try:
        from rdkit import Chem
    except ImportError:
        return {"error": "RDKit is required for FEP ligand validation (pip install rdkit)", "summary": "RDKit is required for FEP ligand validation (pip install rdkit)"}
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None:
        return {"error": f"Invalid SMILES for ligand A: {smiles_a}", "summary": f"Invalid SMILES for ligand A: {smiles_a}"}
    if mol_b is None:
        return {"error": f"Invalid SMILES for ligand B: {smiles_b}", "summary": f"Invalid SMILES for ligand B: {smiles_b}"}
    # Resolve target
    target = _resolve_pdb(target_pdb)
    if "error" in target:
        return {"error": target["error"], "summary": f"FEP failed: {target['error']}"}
    pdb_path = target["path"]

    # Estimate runtime: ~4-8 hours per ligand pair on A100
    estimated_hours = 6.0

    config = {
        "smiles_a": smiles_a,
        "smiles_b": smiles_b,
        "target_pdb": pdb_path,
        "method": method,
        "n_lambda_windows": 12,
        "simulation_time_per_window_ns": 5.0,
        "temperature_k": 300.0,
        "platform": "CUDA",
    }

    from ct.tools.compute import submit_job
    job_result = submit_job(
        job_type="molecular_dynamics",
        params={
            "n_samples": 1,
            "fep_config": config,
        },
        dry_run=kwargs.get("dry_run", True),
    )

    if "error" in job_result:
        return {
            "error": job_result["error"],
            "summary": f"FEP submission failed: {job_result['error']}",
        }

    target_name = Path(pdb_path).stem

    return {
        "summary": (
            f"FEP calculation submitted: {smiles_a[:30]} -> {smiles_b[:30]} "
            f"in {target_name} ({method})"
            + (f" (job: {job_result.get('job_id', 'dry-run')})" if not job_result.get("dry_run") else " [DRY RUN]")
        ),
        "config": config,
        "estimated_hours": estimated_hours,
        "transformation": {"ligand_a": smiles_a, "ligand_b": smiles_b},
        "job": job_result,
    }


# ---------------------------------------------------------------------------
# Binding site / pocket detection
# ---------------------------------------------------------------------------

def _geometric_pocket_detection(pdb_path: str, min_residues: int = 5,
                                 distance_cutoff: float = 8.0) -> list[dict]:
    """Simple geometric pocket detection from PDB coordinates.

    Parses ATOM records, identifies residues with buried atoms (low
    neighbor-averaged solvent exposure), and clusters them by spatial
    proximity.  Returns a list of pocket dicts sorted by size.
    """
    import math

    # Parse atoms
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resname = line[17:20].strip()
                resid = int(line[22:26])
                chain = line[21]
                atoms.append({
                    "x": x, "y": y, "z": z,
                    "resname": resname, "resid": resid, "chain": chain,
                })
            except (ValueError, IndexError):
                continue

    if len(atoms) < 10:
        return []

    # Compute centre of mass
    cx = sum(a["x"] for a in atoms) / len(atoms)
    cy = sum(a["y"] for a in atoms) / len(atoms)
    cz = sum(a["z"] for a in atoms) / len(atoms)

    # For each residue, compute distance to COM and local density
    residue_coords = {}
    for a in atoms:
        key = (a["chain"], a["resid"])
        if key not in residue_coords:
            residue_coords[key] = {"xs": [], "ys": [], "zs": [],
                                    "resname": a["resname"], "chain": a["chain"],
                                    "resid": a["resid"]}
        residue_coords[key]["xs"].append(a["x"])
        residue_coords[key]["ys"].append(a["y"])
        residue_coords[key]["zs"].append(a["z"])

    # Compute residue centres
    residues = []
    for key, rc in residue_coords.items():
        rx = sum(rc["xs"]) / len(rc["xs"])
        ry = sum(rc["ys"]) / len(rc["ys"])
        rz = sum(rc["zs"]) / len(rc["zs"])
        dist_to_com = math.sqrt((rx - cx)**2 + (ry - cy)**2 + (rz - cz)**2)
        residues.append({
            "chain": rc["chain"], "resid": rc["resid"], "resname": rc["resname"],
            "x": rx, "y": ry, "z": rz, "dist_to_com": dist_to_com,
        })

    # Identify cavity residues: not too close to COM (core), not too far (surface)
    dists = [r["dist_to_com"] for r in residues]
    if not dists:
        return []
    median_dist = sorted(dists)[len(dists) // 2]
    cavity_residues = [
        r for r in residues
        if median_dist * 0.4 < r["dist_to_com"] < median_dist * 1.2
    ]

    if len(cavity_residues) < min_residues:
        cavity_residues = sorted(residues, key=lambda r: abs(r["dist_to_com"] - median_dist))[:max(min_residues, len(residues) // 4)]

    # Simple clustering: greedy single-linkage
    clusters = []
    assigned = set()
    for i, res in enumerate(cavity_residues):
        if i in assigned:
            continue
        cluster = [res]
        assigned.add(i)
        for j, other in enumerate(cavity_residues):
            if j in assigned:
                continue
            d = math.sqrt((res["x"] - other["x"])**2 +
                          (res["y"] - other["y"])**2 +
                          (res["z"] - other["z"])**2)
            if d < distance_cutoff:
                cluster.append(other)
                assigned.add(j)
        if len(cluster) >= min_residues:
            clusters.append(cluster)

    # Sort clusters by size descending
    clusters.sort(key=lambda c: len(c), reverse=True)

    pockets = []
    for idx, cluster in enumerate(clusters[:5]):  # top 5 pockets
        xs = [r["x"] for r in cluster]
        ys = [r["y"] for r in cluster]
        zs = [r["z"] for r in cluster]

        # Approximate volume as bounding box volume
        vol = (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs))

        # Druggability heuristic: more residues + larger volume + hydrophobic
        # residues are better
        hydrophobic = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
        n_hydrophobic = sum(1 for r in cluster if r["resname"] in hydrophobic)
        druggability = min(1.0, (len(cluster) / 20) * 0.5 + (n_hydrophobic / max(len(cluster), 1)) * 0.5)

        pockets.append({
            "pocket_id": idx + 1,
            "n_residues": len(cluster),
            "residue_ids": [f"{r['chain']}:{r['resname']}{r['resid']}" for r in cluster],
            "center": {
                "x": round(sum(xs) / len(xs), 2),
                "y": round(sum(ys) / len(ys), 2),
                "z": round(sum(zs) / len(zs), 2),
            },
            "volume_approx_A3": round(vol, 1),
            "druggability_score": round(druggability, 3),
        })

    return pockets


@registry.register(
    name="structure.binding_site",
    description="Detect binding pockets in a protein structure using geometric analysis or fpocket",
    category="structure",
    parameters={
        "pdb_path": "Path to PDB file or UniProt ID for AlphaFold fetch",
        "method": "Detection method: fpocket (default), geometric",
    },
    usage_guide=(
        "You want to find druggable binding pockets in a protein structure. "
        "Use before docking to identify where to focus. fpocket is preferred "
        "if installed; geometric fallback uses coordinate-based clustering."
    ),
)
def binding_site(pdb_path: str, method: str = "fpocket", **kwargs) -> dict:
    """Detect binding pockets in a protein structure.

    * method='fpocket'   — runs fpocket locally if installed, falls back to
                           geometric detection.
    * method='geometric' — pure coordinate-based pocket detection.
    """
    valid_methods = ("fpocket", "geometric")
    if method not in valid_methods:
        return {"error": f"Unknown method '{method}'. Choose from: {', '.join(valid_methods)}", "summary": f"Unknown method '{method}'. Choose from: {', '.join(valid_methods)}"}
    # Resolve PDB path (may be UniProt ID)
    target = _resolve_pdb(pdb_path)
    if "error" in target:
        return {"error": target["error"], "summary": f"Pocket detection failed: {target['error']}"}
    resolved_path = target["path"]
    protein_name = Path(resolved_path).stem

    if method == "fpocket":
        # Try running fpocket
        try:
            proc = subprocess.run(
                ["fpocket", "-f", resolved_path],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                # Parse fpocket output
                out_dir = Path(resolved_path).parent / f"{protein_name}_out"
                info_file = out_dir / f"{protein_name}_info.txt"
                pockets = []

                if info_file.exists():
                    current_pocket = {}
                    for line in info_file.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("Pocket"):
                            if current_pocket:
                                pockets.append(current_pocket)
                            pocket_num = line.split()[1].rstrip(":")
                            current_pocket = {"pocket_id": int(pocket_num)}
                        elif "Score" in line and ":" in line:
                            key, val = line.split(":", 1)
                            try:
                                current_pocket[key.strip().lower().replace(" ", "_")] = float(val.strip())
                            except ValueError:
                                current_pocket[key.strip().lower().replace(" ", "_")] = val.strip()
                        elif "Volume" in line and ":" in line:
                            key, val = line.split(":", 1)
                            try:
                                current_pocket["volume_A3"] = float(val.strip())
                            except ValueError:
                                pass
                    if current_pocket:
                        pockets.append(current_pocket)

                if pockets:
                    top = pockets[0]
                    vol_str = f"vol={top.get('volume_A3', '?')}A^3" if "volume_A3" in top else ""
                    return {
                        "summary": (
                            f"Found {len(pockets)} binding pocket(s) in {protein_name} (fpocket): "
                            f"site 1 ({vol_str})"
                        ),
                        "method": "fpocket",
                        "pockets": pockets,
                    }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Fall through to geometric

    # Geometric fallback
    pockets = _geometric_pocket_detection(resolved_path)

    if not pockets:
        return {
            "summary": f"No binding pockets detected in {protein_name}",
            "method": "geometric",
            "pockets": [],
        }

    top = pockets[0]
    return {
        "summary": (
            f"Found {len(pockets)} binding pocket(s) in {protein_name}: "
            f"site 1 (vol={top['volume_approx_A3']:.0f}A^3, "
            f"druggability={top['druggability_score']:.2f})"
        ),
        "method": "geometric",
        "pockets": pockets,
    }


@registry.register(
    name="structure.immunebuilder",
    description="Predict antibody, TCR, or nanobody structures using ImmuneBuilder",
    category="structure",
    parameters={
        "sequences": "Dictionary of chain sequences (e.g., {'H': '...', 'L': '...'})",
        "model_type": "Type of structure to predict: 'antibody' (default), 'tcr', or 'nanobody'",
        "output_path": "Optional path to save the predicted structure (.pdb)"
    },
    usage_guide=(
        "You want to predict the 3D structure of an antibody, TCR, or nanobody from sequence. "
        "Provide a dictionary of chain sequences (e.g., {'H': '...', 'L': '...'}). "
        "Uses ImmuneBuilder (ABodyBuilder2, TCRBuilder2, NanoBodyBuilder2)."
    )
)
def immunebuilder_predict(sequences: dict, model_type: str = "antibody", output_path: str = None, **kwargs) -> dict:
    """Predict antibody, TCR, or nanobody structures."""
    import tempfile
    
    try:
        from ImmuneBuilder import ABodyBuilder2, TCRBuilder2, NanoBodyBuilder2
    except ImportError:
        return {
            "error": "ImmuneBuilder not installed. Install with: pip install immunebuilder pdbfixer anarci (also requires 'brew install hmmer' or 'conda install hmmer')",
            "summary": "ImmuneBuilder prediction unavailable — dependencies missing",
        }
    
    model_type = model_type.lower()
    if model_type == "antibody":
        predictor = ABodyBuilder2()
    elif model_type == "tcr":
        predictor = TCRBuilder2()
    elif model_type == "nanobody":
        predictor = NanoBodyBuilder2()
    else:
        return {
            "error": f"Invalid model_type '{model_type}'. Choose from 'antibody', 'tcr', 'nanobody'",
            "summary": f"ImmuneBuilder failed — invalid model_type"
        }

    try:
        out = predictor.predict(sequences)
        
        save_path = output_path
        if not save_path:
            save_path = Path(tempfile.mkdtemp(prefix="ct_ib_")) / "prediction.pdb"
            
        out.save(str(save_path))
        
        return {
            "summary": f"Predicted {model_type} structure successfully (saved to {Path(save_path).name})",
            "output_path": str(save_path),
            "model_type": model_type,
            "sequences": sequences
        }
    except Exception as e:
        return {
            "error": f"ImmuneBuilder prediction failed: {e}",
            "summary": f"ImmuneBuilder prediction failed: {e}"
        }
