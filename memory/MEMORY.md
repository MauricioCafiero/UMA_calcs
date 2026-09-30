# Project Memory: UMA Calculations

## Current State

### Project Overview
3D XYZ structure generation from SMILES strings → single-point / geometry-optimization / molecular-dynamics calculations with UMA ML potential → trajectory XYZ with (optional) gradients

### Completed Files

1. **code/smiles_to_xyz.py** - Segment 1: SMILES to 3D XYZ
   - Uses RDKit to parse SMILES, add Hs, embed 3D, optimize with MMFF94
   - Fixed issue with EmbedMolecule params (useRandomCoords on ETKDGv3 object)
   - Working status: Verified

2. **code/run_dynamics.py** - Segment 2: calculations with UMA potential
   - Uses ASE with Fairchem FAIRChemCalculator
   - UMA predictor: `pretrained_mlip.get_predict_unit(checkpoint, device=device)`,
     default checkpoint **uma-s-1p2p1** (cached locally; the old default
     uma-s-1p2 is NOT in the cache)
   - Three modes via `--mode`: `sp` (single point: prints energy + forces),
     `opt` (BFGS/FIRE optimization to `--fmax`, writes optimized structure),
     `md` (Langevin NVT / Verlet NVE, default)
   - `--no-forces` flag: omits forces/gradients from output files (positions only)
   - Reused `write_frames()` helper handles momenta removal + extxyz output
   - **Fixes carried over:**
     - Set `atoms.info['charge'] = 0` and `atoms.info['spin'] = 1` to avoid Fairchem warnings
     - Store forces in `atoms.arrays` (not `atoms.info`) to avoid "ambiguous truth value" comparison error in Fairchem's `check_state()`

3. **requirements.txt** - fairchem-core (brings ase + torch), rdkit

4. **.venv/** - project venv (Python 3.14, created 2026-09-30)

### Project Structure
```
. (repo root)
├── code/
│   ├── smiles_to_xyz.py
│   └── run_dynamics.py
├── results/          # Output directory
├── memory/           # Session persistence
└── README.md
```
(The old MACE training project is preserved, unmodified, in `mace_training_archive/`.)

### Usage
```bash
# Step 1: SMILES to XYZ
python code/smiles_to_xyz.py "CCO" -o results/ethanol.xyz

# Single point
python code/run_dynamics.py results/ethanol.xyz --mode sp

# Geometry optimization
python code/run_dynamics.py results/ethanol.xyz --mode opt -o results/ethanol_opt.xyz

# MD
python code/run_dynamics.py results/ethanol.xyz -o results/trajectory.xyz --steps 1000
```

### Environment Requirements
- `HF_TOKEN` environment variable must be set for HuggingFace

### Last Updated
2026-09-30