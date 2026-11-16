import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from scipy.spatial.distance import cdist


def _seq_cache_key(seq: str) -> str:
    """Return a deterministic short hash key for an amino acid sequence."""
    return hashlib.sha256(seq.strip().encode("utf-8")).hexdigest()[:16]


def _parse_chain_a3m_blocks(a3m_path: str, expected_seqs: list[str]) -> dict[int, list[str]]:
    """
    Parse unpaired per-chain MSA blocks from a ColabFold `.a3m` file.

    Args:
        a3m_path: Path to a ColabFold `complex.a3m` or single-chain `.a3m` file.
        expected_seqs: Ordered chain sequences `[target, lig1, lig2]` or `[single]`.

    Returns:
        Mapping from 0-based chain index to unpadded A3M lines (`>header` and sequence lines).
    """
    with open(a3m_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.rstrip("\r\n") for line in handle if line.strip()]

    if not raw_lines:
        return {}

    chain_lengths = [len(s) for s in expected_seqs]
    start_idx = 0
    if raw_lines[0].startswith("#"):
        header_lens = raw_lines[0][1:].split("\t")[0]
        parsed_lens = [int(x) for x in header_lens.split(",") if x.strip()]
        if parsed_lens:
            chain_lengths = parsed_lens
        start_idx = 1

    # Skip the top combined query header (>101\t102\t103) and concatenated query sequence.
    if start_idx + 1 < len(raw_lines) and raw_lines[start_idx].startswith(">101\t"):
        start_idx += 2

    num_chains = len(chain_lengths)
    chain_tags = {f">{101 + idx}": idx for idx in range(num_chains)}
    per_chain_lines: dict[int, list[str]] = {}
    current_chain: Optional[int] = None
    in_paired_block = True

    for line in raw_lines[start_idx:]:
        if line in chain_tags:
            current_chain = chain_tags[line]
            in_paired_block = False
            per_chain_lines.setdefault(current_chain, []).append(line)
            continue

        if in_paired_block or current_chain is None:
            continue

        if line.startswith(">"):
            per_chain_lines[current_chain].append(line)
            continue

        left_pad = sum(chain_lengths[:current_chain])
        right_pad = sum(chain_lengths[current_chain + 1 :])
        end_slice = len(line) - right_pad if right_pad > 0 else len(line)
        if end_slice > left_pad:
            per_chain_lines[current_chain].append(line[left_pad:end_slice])

    return per_chain_lines


def cache_msas_from_run(
    output_dir: str,
    target_seq: str,
    lig1_seq: str,
    lig2_seq: str,
    cache_dir: str,
) -> bool:
    """
    Cache both the full complex `.a3m` and unpadded per-chain MSAs from a ColabFold run.

    Args:
        output_dir: Directory containing ColabFold outputs (`complex.a3m` or `input.a3m`).
        target_seq: Target chain amino acid sequence.
        lig1_seq: Ligand 1 amino acid sequence.
        lig2_seq: Ligand 2 amino acid sequence.
        cache_dir: Directory where cached `.a3m` files are stored.

    Returns:
        True if at least one MSA file was cached, False otherwise.
    """
    a3m_candidates = [
        os.path.join(output_dir, "complex.a3m"),
        os.path.join(output_dir, "input.a3m"),
    ]
    a3m_path = next((p for p in a3m_candidates if os.path.isfile(p) and os.path.getsize(p) > 0), None)
    if a3m_path is None:
        return False

    os.makedirs(cache_dir, exist_ok=True)
    seqs = [target_seq, lig1_seq, lig2_seq]
    complex_key = _seq_cache_key(":".join(seqs))
    shutil.copy2(a3m_path, os.path.join(cache_dir, f"complex_{complex_key}.a3m"))

    per_chain = _parse_chain_a3m_blocks(a3m_path, seqs)
    for idx, seq in enumerate(seqs):
        lines = per_chain.get(idx)
        if lines:
            chain_file = os.path.join(cache_dir, f"chain_{_seq_cache_key(seq)}.a3m")
            with open(chain_file, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")

    return True


def _assemble_complex_a3m(
    target_seq: str,
    lig1_seq: str,
    lig2_seq: str,
    cache_dir: str,
    dest_a3m_path: str,
) -> bool:
    """
    Assemble a multimer `.a3m` input from cached complex or per-chain `.a3m` files.

    Args:
        target_seq: Target chain amino acid sequence.
        lig1_seq: Ligand 1 amino acid sequence.
        lig2_seq: Ligand 2 amino acid sequence.
        cache_dir: Directory containing cached `.a3m` files.
        dest_a3m_path: Output `.a3m` path for `colabfold_batch`.

    Returns:
        True if a cached `.a3m` was assembled at `dest_a3m_path`, False on cache miss.
    """
    seqs = [target_seq, lig1_seq, lig2_seq]
    complex_key = _seq_cache_key(":".join(seqs))
    cached_complex = os.path.join(cache_dir, f"complex_{complex_key}.a3m")
    if os.path.isfile(cached_complex) and os.path.getsize(cached_complex) > 0:
        shutil.copy2(cached_complex, dest_a3m_path)
        return True

    chain_files = [os.path.join(cache_dir, f"chain_{_seq_cache_key(seq)}.a3m") for seq in seqs]
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in chain_files):
        return False

    lengths = [len(seq) for seq in seqs]
    out_lines = [
        f"#{','.join(str(x) for x in lengths)}\t1,1,1",
        ">101\t102\t103",
        "".join(seqs),
    ]

    for idx, chain_file in enumerate(chain_files):
        left_pad = "-" * sum(lengths[:idx])
        right_pad = "-" * sum(lengths[idx + 1 :])
        chain_tag = f">{101 + idx}"
        with open(chain_file, "r", encoding="utf-8") as handle:
            lines = [line.rstrip("\r\n") for line in handle if line.strip()]

        for line in lines:
            if line.startswith(">101") or line.startswith(">102") or line.startswith(">103"):
                out_lines.append(chain_tag)
            elif line.startswith(">"):
                out_lines.append(line)
            else:
                out_lines.append(f"{left_pad}{line}{right_pad}")

    with open(dest_a3m_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out_lines) + "\n")
    return True


def _fetch_missing_chain_msas(
    seqs: list[str],
    cache_dir: str,
    env: dict[str, str],
) -> None:
    """
    Fetch and cache single-chain MSAs via `colabfold_batch --msa-only` when some chains
    are already cached (e.g., cached target across tournament matches).
    Silently falls back if `--msa-only` is unavailable or fails.
    """
    unique_seqs = list(dict.fromkeys(seqs))
    cached_flags = [
        os.path.isfile(os.path.join(cache_dir, f"chain_{_seq_cache_key(s)}.a3m"))
        for s in unique_seqs
    ]
    # Only pre-fetch individual missing chains if at least one chain is already cached;
    # otherwise a standard multimer FASTA run fetches all chains in one request.
    if not any(cached_flags) or all(cached_flags):
        return

    missing_seqs = [s for s, is_cached in zip(unique_seqs, cached_flags) if not is_cached]
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp_dir:
        fasta_path = os.path.join(tmp_dir, "missing.fasta")
        msa_out_dir = os.path.join(tmp_dir, "msa_out")
        os.makedirs(msa_out_dir, exist_ok=True)
        with open(fasta_path, "w", encoding="utf-8") as handle:
            for idx, seq in enumerate(missing_seqs):
                handle.write(f">seq_{idx}\n{seq}\n")

        cmd = ["colabfold_batch", fasta_path, msa_out_dir, "--msa-only"]
        try:
            subprocess.run(
                cmd,
                shell=False,
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, OSError):
            return

        for idx, seq in enumerate(missing_seqs):
            a3m_file = os.path.join(msa_out_dir, f"seq_{idx}.a3m")
            if os.path.isfile(a3m_file) and os.path.getsize(a3m_file) > 0:
                per_chain = _parse_chain_a3m_blocks(a3m_file, [seq])
                lines = per_chain.get(0)
                if lines:
                    dest = os.path.join(cache_dir, f"chain_{_seq_cache_key(seq)}.a3m")
                    with open(dest, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")


def _setup_colabfold_run(
    target_seq: str,
    lig1_seq: str,
    lig2_seq: str,
    base_output_dir: str,
    run_name: str,
    num_seeds: int = 4,
    num_recycles: int = 10,
    num_models: int = 5,
    use_msa_cache: bool = True,
    msa_cache_dir: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> tuple[list[str], str, dict[str, str]]:
    """
    Prepare output directory, input sequence/MSA file, and `colabfold_batch` command.

    Defaults use 4 seeds x 5 models (20 predictions) with a 10-recycle cap and
    automated `.a3m` MSA caching without altering template settings.
    """
    actual_output_dir = os.path.join(base_output_dir, run_name)
    counter = 1
    while os.path.exists(actual_output_dir) and os.listdir(actual_output_dir):
        actual_output_dir = os.path.join(base_output_dir, f"{run_name}_{counter}")
        counter += 1

    os.makedirs(actual_output_dir, exist_ok=True)
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)

    fasta_path = os.path.join(actual_output_dir, "input.fasta")
    complex_seq = f"{target_seq}:{lig1_seq}:{lig2_seq}"
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write(f">complex\n{complex_seq}\n")

    input_path = fasta_path
    if use_msa_cache:
        resolved_cache_dir = msa_cache_dir or os.path.join(base_output_dir, ".msa_cache")
        a3m_input_path = os.path.join(actual_output_dir, "input.a3m")
        if not _assemble_complex_a3m(target_seq, lig1_seq, lig2_seq, resolved_cache_dir, a3m_input_path):
            _fetch_missing_chain_msas([target_seq, lig1_seq, lig2_seq], resolved_cache_dir, env)
            if _assemble_complex_a3m(target_seq, lig1_seq, lig2_seq, resolved_cache_dir, a3m_input_path):
                input_path = a3m_input_path
        else:
            input_path = a3m_input_path

    random_seed = random.randint(1, 999999)
    cmd = [
        "colabfold_batch",
        input_path,
        actual_output_dir,
        "--random-seed",
        str(random_seed),
        "--num-seeds",
        str(num_seeds),
        "--num-models",
        str(num_models),
        "--use-dropout",
        "--num-recycle",
        str(num_recycles),
    ]
    if extra_args:
        cmd.extend(extra_args)

    return cmd, actual_output_dir, env


def run_colabfold(
    target_seq: str,
    lig1_seq: str,
    lig2_seq: str,
    base_output_dir: str,
    run_name: str = "competition",
    num_seeds: int = 4,
    num_recycles: int = 10,
    num_models: int = 5,
    use_msa_cache: bool = True,
    msa_cache_dir: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> str:
    """
    Run `colabfold_batch` synchronously and cache generated MSAs on success.
    """
    cmd, actual_output_dir, env = _setup_colabfold_run(
        target_seq=target_seq,
        lig1_seq=lig1_seq,
        lig2_seq=lig2_seq,
        base_output_dir=base_output_dir,
        run_name=run_name,
        num_seeds=num_seeds,
        num_recycles=num_recycles,
        num_models=num_models,
        use_msa_cache=use_msa_cache,
        msa_cache_dir=msa_cache_dir,
        extra_args=extra_args,
    )

    log_path = os.path.join(actual_output_dir, "colabfold.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                cmd,
                shell=False,
                check=True,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            print(f"ColabFold execution failed. Command run was:\n{' '.join(cmd)}")
            print(f"Check {log_path} for detailed error output.")
            raise

    if use_msa_cache:
        resolved_cache_dir = msa_cache_dir or os.path.join(base_output_dir, ".msa_cache")
        cache_msas_from_run(actual_output_dir, target_seq, lig1_seq, lig2_seq, resolved_cache_dir)

    return actual_output_dir


def run_colabfold_async(
    target_seq: str,
    lig1_seq: str,
    lig2_seq: str,
    base_output_dir: str,
    run_name: str = "competition",
    num_seeds: int = 4,
    num_recycles: int = 10,
    num_models: int = 5,
    use_msa_cache: bool = True,
    msa_cache_dir: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
):
    """
    Launch `colabfold_batch` asynchronously and return `(process, output_dir, log_file)`.
    """
    cmd, actual_output_dir, env = _setup_colabfold_run(
        target_seq=target_seq,
        lig1_seq=lig1_seq,
        lig2_seq=lig2_seq,
        base_output_dir=base_output_dir,
        run_name=run_name,
        num_seeds=num_seeds,
        num_recycles=num_recycles,
        num_models=num_models,
        use_msa_cache=use_msa_cache,
        msa_cache_dir=msa_cache_dir,
        extra_args=extra_args,
    )

    log_path = os.path.join(actual_output_dir, "colabfold.log")
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd, shell=False, env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    return process, actual_output_dir, log_file


def extract_model_id(pdb_filename: str) -> str:
    """
    Extract a stable `(model, seed)` identifier from a ColabFold PDB filename,
    ignoring `rank_NNN_` prefixes added when ColabFold finalizes a run.
    """
    match = re.search(r"(model_\d+_seed_\d+)", pdb_filename)
    if match:
        return match.group(1)
    seed_match = re.search(r"(seed_\d+)", pdb_filename)
    return seed_match.group(1) if seed_match else pdb_filename


def _mean_chain_plddt(chain) -> Optional[float]:
    """Return mean B-factor (pLDDT) across atoms of a Bio.PDB Chain, or None if empty."""
    b_factors = [atom.get_bfactor() for atom in chain.get_atoms()]
    return float(np.mean(b_factors)) if b_factors else None


def _load_colabfold_scores(pdb_path: str) -> tuple[Optional[float], Optional[float]]:
    """Load `(ptm, iptm)` from the companion ColabFold `scores_*.json` file if present."""
    pdb_file = Path(pdb_path)
    candidates = [
        re.sub(r"_(?:un)?relaxed_", "_scores_", pdb_file.name),
        re.sub(r"_(?:un)?relaxed_rank_\d+_", "_scores_", pdb_file.name),
    ]
    for name in candidates:
        score_path = pdb_file.with_name(Path(name).with_suffix(".json").name)
        if score_path.is_file():
            try:
                with open(score_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                ptm = float(data["ptm"]) if "ptm" in data and data["ptm"] is not None else None
                iptm = float(data["iptm"]) if "iptm" in data and data["iptm"] is not None else None
                return ptm, iptm
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return None, None
    return None, None


def calculate_interface_distance(
    structure, target_residues: list[int], chain_id_ligand: str, top_n: int = 5
) -> float:
    """
    Calculates the average distance of the top N closest heavy-atom pairs
    between specified target residues' side chains (Chain A) and a ligand chain.
    Falls back to CA for Glycine or residues without side chain heavy atoms.
    """
    try:
        model = structure[0]
        chain_a = model["A"]
        chain_lig = model[chain_id_ligand]
    except KeyError:
        return float("inf")

    target_atoms = []
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}

    for res_id in target_residues:
        if res_id in chain_a:
            residue = chain_a[res_id]
            res_sidechain_atoms = []
            for atom in residue:
                # Exclude hydrogens and backbone atoms to get side-chain heavy atoms
                if atom.element != "H" and atom.get_name() not in backbone_atoms:
                    res_sidechain_atoms.append(atom.get_coord())

            if not res_sidechain_atoms:
                # Fallback to CA if no side chain heavy atoms exist (e.g., Glycine)
                if "CA" in residue:
                    res_sidechain_atoms.append(residue["CA"].get_coord())

            target_atoms.extend(res_sidechain_atoms)

    if not target_atoms:
        return float("inf")

    lig_atoms = []
    for residue in chain_lig:
        for atom in residue:
            if atom.element != "H":
                lig_atoms.append(atom.get_coord())

    if not lig_atoms:
        return float("inf")

    target_coords = np.array(target_atoms)
    lig_coords = np.array(lig_atoms)

    # Calculate pairwise distances
    distances = cdist(target_coords, lig_coords)

    # Flatten the distances
    flat_distances = distances.flatten()

    # Take the average of the top_n shortest distances
    n_to_take = min(top_n, len(flat_distances))
    if n_to_take == 0:
        return float("inf")
        
    if n_to_take < len(flat_distances):
        top_distances = np.partition(flat_distances, n_to_take - 1)[:n_to_take]
    else:
        top_distances = flat_distances
        
    return float(np.mean(top_distances))


def analyze_binding(
    pdb_path: str,
    target_residues: list[int],
    lig1_seq: str,
    lig2_seq: str,
    distance_threshold: float = 10.0,
) -> dict:
    """
    Parse a PDB file and determine ligand binding and confidence metrics.

    `mean_plddt` averages B-factors over the target chain (`A`) and the closer
    (active) ligand chain so a displaced unbound competitor does not penalize valid
    single-bound complexes. The full 3-chain atom average is preserved as
    `complex_plddt`, alongside per-chain pLDDTs and companion `ptm`/`iptm` scores.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    all_b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
    complex_plddt = float(np.mean(all_b_factors)) if all_b_factors else 0.0

    ppb = PPBuilder()
    model = structure[0]

    chain_lig1 = "B"
    chain_lig2 = "C"
    assigned_lig1 = False
    assigned_lig2 = False

    for chain in model:
        chain_id = chain.get_id()
        if chain_id == "A":
            continue
        peptides = ppb.build_peptides(chain)
        if peptides:
            seq = str(peptides[0].get_sequence())
            if seq == lig1_seq and not assigned_lig1:
                chain_lig1 = chain_id
                assigned_lig1 = True
            elif seq == lig2_seq and not assigned_lig2:
                chain_lig2 = chain_id
                assigned_lig2 = True

    dist_lig1 = calculate_interface_distance(structure, target_residues, chain_lig1)
    dist_lig2 = calculate_interface_distance(structure, target_residues, chain_lig2)

    plddt_target = _mean_chain_plddt(model["A"]) if "A" in model else None
    plddt_lig1 = _mean_chain_plddt(model[chain_lig1]) if chain_lig1 in model else None
    plddt_lig2 = _mean_chain_plddt(model[chain_lig2]) if chain_lig2 in model else None

    active_chain = chain_lig1 if dist_lig1 <= dist_lig2 else chain_lig2
    plddt_active_lig = plddt_lig1 if dist_lig1 <= dist_lig2 else plddt_lig2

    if "A" in model and active_chain in model:
        bound_b_factors = [
            atom.get_bfactor()
            for chain_key in ("A", active_chain)
            for atom in model[chain_key].get_atoms()
        ]
        mean_plddt = float(np.mean(bound_b_factors)) if bound_b_factors else complex_plddt
    else:
        mean_plddt = complex_plddt

    ptm, iptm = _load_colabfold_scores(pdb_path)

    return {
        "pdb_file": os.path.basename(pdb_path),
        "model_id": extract_model_id(os.path.basename(pdb_path)),
        "dist_lig1": dist_lig1,
        "dist_lig2": dist_lig2,
        "lig1_bound": dist_lig1 < distance_threshold,
        "lig2_bound": dist_lig2 < distance_threshold,
        "mean_plddt": mean_plddt,
        "complex_plddt": complex_plddt,
        "plddt_target": plddt_target,
        "plddt_lig1": plddt_lig1,
        "plddt_lig2": plddt_lig2,
        "plddt_active_lig": plddt_active_lig,
        "ptm": ptm,
        "iptm": iptm,
        "chain_lig1": chain_lig1,
        "chain_lig2": chain_lig2,
    }


def process_ensemble(
    output_dir: str,
    target_residues: list[int],
    lig1_seq: str,
    lig2_seq: str,
    distance_threshold: float = 10.0,
) -> list[dict]:
    """
    Analyzes an ensemble of PDB files in a directory.
    """
    results = []
    pdb_files = sorted(Path(output_dir).glob("*.pdb"))

    for pdb in pdb_files:
        result = analyze_binding(str(pdb), target_residues, lig1_seq, lig2_seq, distance_threshold)
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
    if step <= 0:
        raise ValueError("step must be positive")

    best_discrimination = -1
    best_thresh = min_thresh
    best_stats = {}

    n_steps = int(round((max_thresh - min_thresh) / step))
    for i in range(n_steps + 1):
        current_thresh = round(min_thresh + i * step, 6)
        if current_thresh > max_thresh + 1e-9:
            break
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

    return {
        "optimal_threshold": round(best_thresh, 2),
        "max_discrimination": best_discrimination,
        "stats": best_stats,
    }
