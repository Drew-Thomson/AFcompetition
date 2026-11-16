import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest
from Bio.PDB import PDBParser

from af_competition import (
    _setup_colabfold_run,
    analyze_binding,
    cache_msas_from_run,
    calculate_interface_distance,
    extract_model_id,
    optimize_threshold,
)


@pytest.fixture
def mock_pdb_file(tmp_path):
    # Create a minimal PDB file for testing
    # Chain A: Target (Residue 1 and 2)
    # Chain B: Ligand 1 (close to Residue 1)
    # Chain C: Ligand 2 (far from Residue 1 and 2)

    pdb_content = (
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 10.00           C  \n"
        "ATOM      2  CB  ALA A   1      10.500  10.500  10.500  1.00 10.00           C  \n"  # Sidechain
        "ATOM      3  CA  ALA A   2      12.000  12.000  12.000  1.00 10.00           C  \n"
        "ATOM      4  CA  GLY B   1      11.000  11.000  11.000  1.00 10.00           C  \n"  # close to A:1 CB
        "ATOM      5  CA  GLY C   1      50.000  50.000  50.000  1.00 10.00           C  \n"  # > 40A from A
    )
    pdb_path = os.path.join(tmp_path, "mock_complex.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    return pdb_path


def test_analyze_binding(mock_pdb_file):
    target_residues = [1]
    result = analyze_binding(mock_pdb_file, target_residues, "G", "G", distance_threshold=10.0)

    assert result["lig1_bound"] is True
    assert result["lig2_bound"] is False
    assert result["dist_lig1"] < 2.0
    assert result["dist_lig2"] > 35.0


def test_missing_chain_returns_inf(mock_pdb_file):
    target_residues = [1]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", mock_pdb_file)

    dist = calculate_interface_distance(structure, target_residues, "D")
    assert dist == float("inf")


def test_analyze_binding_excludes_unbound_competitor_from_mean_plddt(tmp_path):
    pdb_content = (
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 92.00           C  \n"
        "ATOM      2  CB  ALA A   1      10.500  10.500  10.500  1.00 92.00           C  \n"
        "ATOM      3  CA  GLY B   1      11.000  11.000  11.000  1.00 88.00           C  \n"
        "ATOM      4  CA  GLY C   1      50.000  50.000  50.000  1.00 40.00           C  \n"
    )
    pdb_path = tmp_path / "complex_unrelaxed_rank_001_alphafold2_multimer_v3_model_2_seed_123.pdb"
    pdb_path.write_text(pdb_content)

    score_path = tmp_path / "complex_scores_rank_001_alphafold2_multimer_v3_model_2_seed_123.json"
    score_path.write_text(json.dumps({"ptm": 0.82, "iptm": 0.77}))

    result = analyze_binding(str(pdb_path), [1], "G", "G", distance_threshold=5.0)

    assert result["model_id"] == "model_2_seed_123"
    assert result["plddt_target"] == pytest.approx(92.0)
    assert result["plddt_lig1"] == pytest.approx(88.0)
    assert result["plddt_lig2"] == pytest.approx(40.0)
    assert result["plddt_active_lig"] == pytest.approx(88.0)
    # Mean over Target (2 atoms @ 92) + active Lig1 (1 atom @ 88) -> 90.67, not penalized by Lig2 @ 40
    assert result["mean_plddt"] == pytest.approx((92.0 * 2 + 88.0) / 3)
    assert result["complex_plddt"] == pytest.approx((92.0 * 2 + 88.0 + 40.0) / 4)
    assert result["ptm"] == pytest.approx(0.82)
    assert result["iptm"] == pytest.approx(0.77)


def test_extract_model_id_distinguishes_models_and_ranks():
    unranked = "complex_unrelaxed_alphafold2_multimer_v3_model_3_seed_99.pdb"
    ranked = "complex_unrelaxed_rank_004_alphafold2_multimer_v3_model_3_seed_99.pdb"
    other_model = "complex_unrelaxed_rank_001_alphafold2_multimer_v3_model_5_seed_99.pdb"

    assert extract_model_id(unranked) == "model_3_seed_99"
    assert extract_model_id(ranked) == "model_3_seed_99"
    assert extract_model_id(other_model) == "model_5_seed_99"


def test_setup_colabfold_run_defaults_and_no_templates(tmp_path):
    cmd, out_dir, _ = _setup_colabfold_run(
        target_seq="ACDE",
        lig1_seq="FG",
        lig2_seq="HI",
        base_output_dir=str(tmp_path),
        run_name="screen",
        use_msa_cache=False,
    )

    assert os.path.isdir(out_dir)
    assert "--num-seeds" in cmd and cmd[cmd.index("--num-seeds") + 1] == "4"
    assert "--num-models" in cmd and cmd[cmd.index("--num-models") + 1] == "5"
    assert "--num-recycle" in cmd and cmd[cmd.index("--num-recycle") + 1] == "10"
    assert "--use-dropout" in cmd
    assert "--templates" not in cmd
    assert "--custom-template-path" not in cmd


def test_msa_caching_round_trip_and_partial_hit(tmp_path, monkeypatch):
    # Simulate completed Match 1 run with target='ACDE', lig1='FG', lig2='HI'
    match1_dir = tmp_path / "match1"
    match1_dir.mkdir()
    complex_a3m = "\n".join(
        [
            "#4,2,2\t1,1,1",
            ">101\t102\t103",
            "ACDEFGHI",
            ">101",
            "ACDE----",
            ">UniRef_target_hit",
            "ACaaDE----",
            ">102",
            "----FG--",
            ">UniRef_lig1_hit",
            "----YG--",
            ">103",
            "------HI",
            "",
        ]
    )
    (match1_dir / "complex.a3m").write_text(complex_a3m)

    cache_dir = tmp_path / ".msa_cache"
    assert cache_msas_from_run(str(match1_dir), "ACDE", "FG", "HI", str(cache_dir))

    # Match 2 introduces a new challenger 'KLM', keeping target 'ACDE' and champion 'FG'.
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        fasta_in = cmd[1]
        out_msa_dir = cmd[2]
        content = open(fasta_in).read()
        assert "ACDE" not in content
        assert "FG" not in content
        assert "KLM" in content
        with open(os.path.join(out_msa_dir, "seq_0.a3m"), "w") as handle:
            handle.write("#3\t1\n>101\nKLM\n>UniRef_lig3_hit\nKVM\n")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    cmd, match2_dir, _ = _setup_colabfold_run(
        target_seq="ACDE",
        lig1_seq="FG",
        lig2_seq="KLM",
        base_output_dir=str(tmp_path),
        run_name="match2",
        use_msa_cache=True,
    )

    assert len(calls) == 1
    assert "--msa-only" in calls[0]
    assert cmd[1].endswith("input.a3m")

    assembled = open(os.path.join(match2_dir, "input.a3m")).read().splitlines()
    assert assembled[0] == "#4,2,3\t1,1,1"
    assert assembled[2] == "ACDEFGKLM"
    assert "ACaaDE-----" in assembled
    assert "----YG---" in assembled
    assert "------KVM" in assembled


def test_optimize_threshold_includes_max_bound():
    results = [
        {"pdb_file": "a.pdb", "dist_lig1": 7.9, "dist_lig2": 12.0},
        {"pdb_file": "b.pdb", "dist_lig1": 12.0, "dist_lig2": 7.9},
    ]
    opt = optimize_threshold(results, min_thresh=2.0, max_thresh=8.0, step=0.25)
    assert opt["optimal_threshold"] == 8.0
    assert opt["max_discrimination"] == 2
