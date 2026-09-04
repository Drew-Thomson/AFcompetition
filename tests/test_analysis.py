import os
import pytest
from af_competition import analyze_binding, calculate_ca_distance


@pytest.fixture
def mock_pdb_file(tmp_path):
    # Create a minimal PDB file for testing
    # Chain A: Target (Residue 1 and 2)
    # Chain B: Ligand 1 (close to Residue 1)
    # Chain C: Ligand 2 (far from Residue 1 and 2)

    pdb_content = (
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 10.00           C  \n"
        "ATOM      2  CA  ALA A   2      12.000  12.000  12.000  1.00 10.00           C  \n"
        "ATOM      3  CA  GLY B   1      11.000  11.000  11.000  1.00 10.00           C  \n"  # ~1.7A from A:1
        "ATOM      4  CA  GLY C   1      50.000  50.000  50.000  1.00 10.00           C  \n"  # > 40A from A
    )
    pdb_path = os.path.join(tmp_path, "mock_complex.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    return pdb_path


def test_analyze_binding(mock_pdb_file):
    target_residues = [1]
    result = analyze_binding(mock_pdb_file, target_residues, distance_threshold=10.0)

    assert result["lig1_bound"] is True
    assert result["lig2_bound"] is False
    assert result["dist_lig1"] < 2.0
    assert result["dist_lig2"] > 40.0


def test_missing_chain_returns_inf(mock_pdb_file):
    target_residues = [1]

    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", mock_pdb_file)

    dist = calculate_ca_distance(structure, target_residues, "D")
    assert dist == float("inf")
