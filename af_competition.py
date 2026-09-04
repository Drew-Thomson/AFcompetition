import os
import subprocess
import random
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser


def run_colabfold(
    target_seq: str,
    lig1_seq: str,
    lig2_seq: str,
    base_output_dir: str,
    run_name: str = "competition",
    num_seeds: int = 20,
) -> str:
    """
    Executes colabfold_batch for the provided target and ligand sequences.

    Args:
        target_seq: Amino acid sequence of the target protein.
        lig1_seq: Amino acid sequence of ligand 1.
        lig2_seq: Amino acid sequence of ligand 2.
        base_output_dir: Base directory to save colabfold output.
        run_name: Name of the specific run, used for the subfolder.
        num_seeds: Number of models to generate in the ensemble.

    Returns:
        The path to the generated output directory.
    """
    actual_output_dir = os.path.join(base_output_dir, run_name)
    counter = 1
    while os.path.exists(actual_output_dir) and os.listdir(actual_output_dir):
        # Only increment if the directory exists AND is not empty
        actual_output_dir = os.path.join(base_output_dir, f"{run_name}_{counter}")
        counter += 1

    os.makedirs(actual_output_dir, exist_ok=True)
    fasta_path = os.path.join(actual_output_dir, "input.fasta")

    complex_seq = f"{target_seq}:{lig1_seq}:{lig2_seq}"

    with open(fasta_path, "w") as f:
        f.write(f">complex\n{complex_seq}\n")

    random_seed = random.randint(1, 999999)

    cmd = [
        "colabfold_batch",
        fasta_path,
        actual_output_dir,
        "--random-seed",
        str(random_seed),
        "--num-seeds",
        str(num_seeds),
        "--num-models",
        "1",
        "--use-dropout",
        "--num-recycle",
        "20",
    ]

    # Convert to a single string and run with shell=True to ensure any local shell aliases/paths are respected
    cmd_str = " ".join(cmd)

    # Sanitize environment variables to prevent Jupyter's MPLBACKEND from breaking colabfold_batch
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)

    try:
        subprocess.run(cmd_str, shell=True, check=True, env=env)
    except subprocess.CalledProcessError:
        print(f"ColabFold execution failed. Command run was:\n{cmd_str}")
        raise

    return actual_output_dir


def calculate_ca_distance(
    structure, target_residues: list[int], chain_id_ligand: str
) -> float:
    """
    Calculates the minimum CA-CA distance between specified target residues (Chain A) and a ligand chain.
    """
    try:
        model = structure[0]
        chain_a = model["A"]
        chain_lig = model[chain_id_ligand]
    except KeyError:
        return float("inf")

    target_ca_atoms = []
    for res_id in target_residues:
        if res_id in chain_a:
            residue = chain_a[res_id]
            if "CA" in residue:
                target_ca_atoms.append(residue["CA"].get_coord())

    if not target_ca_atoms:
        return float("inf")

    lig_ca_atoms = []
    for residue in chain_lig:
        if "CA" in residue:
            lig_ca_atoms.append(residue["CA"].get_coord())

    if not lig_ca_atoms:
        return float("inf")

    target_coords = np.array(target_ca_atoms)
    lig_coords = np.array(lig_ca_atoms)

    # Calculate pairwise distances and return the minimum
    # shape of target_coords is (N, 3), lig_coords is (M, 3)
    diff = target_coords[:, np.newaxis, :] - lig_coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return float(np.min(distances))


def analyze_binding(
    pdb_path: str, target_residues: list[int], distance_threshold: float = 10.0
) -> dict:
    """
    Parses a PDB file and determines which ligands are bound based on CA-CA distance.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
    mean_plddt = float(np.mean(b_factors)) if b_factors else 0.0

    dist_lig1 = calculate_ca_distance(structure, target_residues, "B")
    dist_lig2 = calculate_ca_distance(structure, target_residues, "C")

    return {
        "pdb_file": os.path.basename(pdb_path),
        "dist_lig1": dist_lig1,
        "dist_lig2": dist_lig2,
        "lig1_bound": dist_lig1 < distance_threshold,
        "lig2_bound": dist_lig2 < distance_threshold,
        "mean_plddt": mean_plddt,
    }


def process_ensemble(
    output_dir: str, target_residues: list[int], distance_threshold: float = 10.0
) -> list[dict]:
    """
    Analyzes an ensemble of PDB files in a directory.
    """
    results = []
    pdb_files = list(Path(output_dir).glob("*.pdb"))

    for pdb in pdb_files:
        result = analyze_binding(str(pdb), target_residues, distance_threshold)
        results.append(result)

    return results


def optimize_threshold(
    results: list[dict],
    min_thresh: float = 5.0,
    max_thresh: float = 15.0,
    step: float = 0.5,
) -> dict:
    """
    Finds the distance threshold that maximizes the number of singly bound states.
    """
    best_discrimination = -1
    best_thresh = min_thresh
    best_stats = {}

    current_thresh = min_thresh
    while current_thresh <= max_thresh:
        lig1_results = [
            r
            for r in results
            if r["dist_lig1"] < current_thresh and r["dist_lig2"] >= current_thresh
        ]
        lig2_results = [
            r
            for r in results
            if r["dist_lig2"] < current_thresh and r["dist_lig1"] >= current_thresh
        ]
        both_results = [
            r
            for r in results
            if r["dist_lig1"] < current_thresh and r["dist_lig2"] < current_thresh
        ]
        neither_results = [
            r
            for r in results
            if r["dist_lig1"] >= current_thresh and r["dist_lig2"] >= current_thresh
        ]

        discrimination = len(lig1_results) + len(lig2_results)

        if discrimination > best_discrimination:
            best_discrimination = discrimination
            best_thresh = current_thresh
            best_stats = {
                "lig1_wins": len(lig1_results),
                "lig1_files": [r["pdb_file"] for r in lig1_results],
                "lig1_results": lig1_results,
                "lig2_wins": len(lig2_results),
                "lig2_files": [r["pdb_file"] for r in lig2_results],
                "lig2_results": lig2_results,
                "both_bound": len(both_results),
                "both_files": [r["pdb_file"] for r in both_results],
                "both_results": both_results,
                "neither_bound": len(neither_results),
                "neither_files": [r["pdb_file"] for r in neither_results],
                "neither_results": neither_results,
            }

        current_thresh += step

    return {
        "optimal_threshold": round(best_thresh, 2),
        "max_discrimination": best_discrimination,
        "stats": best_stats,
    }
