# Molecular Calculations with UMA ML Potential

This project converts SMILES strings to 3D structures and runs single-point,
geometry-optimization or molecular-dynamics calculations with the UMA ML
potential, producing trajectory files with or without gradients.

## Overview

The pipeline consists of two segments:

1. **SMILES to XYZ** (`code/smiles_to_xyz.py`): Converts a SMILES string to a 3D XYZ file using RDKit
2. **Calculations** (`code/run_dynamics.py`): Single-point, geometry optimization and MD using ASE with the Fairchem UMA calculator

## Requirements

- Python 3.12+
- Fairchem (fairchem-core), which brings ASE and PyTorch with it
- RDKit

See `requirements.txt`.

## Installation

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

After activation, plain `python` in the commands below refers to the
environment's interpreter.

## Setup

Set your HuggingFace API token as an environment variable. Create a token
at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens);
your account must also have accepted the license on the
[UMA model page](https://huggingface.co/facebook/UMA) (the weights are gated).

```bash
export HF_TOKEN="your_token_here"
```

On Windows:
```cmd
set HF_TOKEN=your_token_here
```

On the first run the UMA checkpoint (~2 GB) is downloaded once from
HuggingFace and cached locally (`~/.cache/fairchem`, or the HuggingFace
cache); later runs use the cached copy, including offline. If the token
is missing or the license has not been accepted, the download fails with
a permission error. The default checkpoint is `uma-s-1p2p1`.

## Usage

### Step 1: Generate 3D Structure from SMILES

```bash
python code/smiles_to_xyz.py "CCO" -o results/ethanol.xyz
```

This will:
- Parse the SMILES string
- Add hydrogens
- Embed in 3D and optimize geometry with MMFF94
- Save to `results/ethanol.xyz`

### Step 2: Run Calculations

**Single point** — prints the energy and forces:

```bash
python code/run_dynamics.py results/ethanol.xyz --mode sp
```

**Geometry optimization** — writes the optimized structure:

```bash
python code/run_dynamics.py results/ethanol.xyz --mode opt -o results/ethanol_opt.xyz --fmax 0.01
```

Options:
- `--fmax`: Force convergence criterion in eV/A (default: 0.01)
- `--max-steps`: Maximum optimizer steps (default: 500)
- `--optimizer`: `bfgs` or `fire` (default: bfgs)

**Molecular dynamics** — writes a trajectory:

```bash
python code/run_dynamics.py results/ethanol.xyz -o results/trajectory.xyz --steps 1000 --temperature 300
```

Options (`md` mode, the default):
- `--steps`: Number of MD steps (default: 1000)
- `--timestep`: Timestep in femtoseconds (default: 1.0)
- `--temperature`: Temperature in Kelvin (default: 300)
- `--md-type`: `langevin` (NVT) or `verlet` (NVE) (default: langevin)

**Options for all modes:**
- `--no-forces`: Do not write forces/gradients to the output file
- `--checkpoint`: UMA checkpoint name (default: uma-s-1p2p1)

### Full Pipeline Example

```bash
# Ethanol example
python code/smiles_to_xyz.py "CCO" -o results/ethanol.xyz
python code/run_dynamics.py results/ethanol.xyz -o results/ethanol_trajectory.xyz --steps 500 --temperature 300
```

## Output

The final trajectory file contains:
- All frames from the MD simulation
- Atomic forces (gradients) for each frame in extended XYZ format,
  unless `--no-forces` is given (positions only)

## Project Structure

```
.
├── code/
│   ├── smiles_to_xyz.py    # Segment 1: SMILES to 3D XYZ
│   └── run_dynamics.py     # Segment 2: single point / optimization / MD
├── results/                 # Output files
├── requirements.txt
└── README.md
```