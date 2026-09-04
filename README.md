# AFcompetition

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![pytest](https://img.shields.io/badge/pytest-passing-success.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

AlphaFold Ensemble Competition Screen.

Automated pipeline for the methodology described in:
*AlphaFold Ensemble Competition Screens Enable Peptide Binder Design with Single-Residue Sensitivity* (DOI: 10.1021/acschembio.4c00418).

## Prerequisites

* `colabfold_batch` must be installed and accessible via the system path.
* Python dependencies are listed in `requirements.txt`. Install via:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

Execution is managed via the `AF_Competition_Screen.ipynb` Jupyter Notebook.

User-configurable parameters include:
* `TARGET_SEQ`, `LIGAND_1_SEQ`, `LIGAND_2_SEQ`: Amino acid sequences.
* `BINDING_SITE_RESIDUES`: Target chain residue indices defining the binding site (1-indexed).
* `NUM_SEEDS`: Number of structural models to generate.
* `RUN_NAME`: Subdirectory name for output isolation.
* `MIN_PLDDT`: Minimum mean pLDDT score for a model to be included in analysis.

## Pipeline Operations

1. **Generation**: Runs `colabfold_batch` to generate model ensembles in isolated subdirectories.
2. **Analysis**: Filters generated models by `MIN_PLDDT`. Optimizes a CA-CA distance threshold (5.0 Å to 15.0 Å) to maximize discrimination of singly bound states. 
3. **Visualization**: Tallies binding states, lists output files for each state, and renders the highest-confidence winning structure inline using `py3Dmol`.

## Development & Testing

Tests are implemented with `pytest`. Run tests via:
```bash
PYTHONPATH=. pytest tests/
```

Code styling is enforced with `ruff`. Run linting via:
```bash
ruff check .
```

## License

Distributed under the MIT License. See `LICENSE` for more information.