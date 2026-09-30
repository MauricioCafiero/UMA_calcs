# CLAUDE.md

## Project Overview
This code generates 3D XYZ structures from SMILES strings and runs single-point,
geometry-optimization or molecular-dynamics calculations on them with the UMA
ML potential, producing an XYZ file with or without gradients.

## Tech Stack
- Language: Python
- packages:
  * RDKit for SMILES → 3D structure generation
  * ASE for running dynamics/optimizations and generating the XYZ trajectory file
  * Fairchem for the ASE calculator: use the UMA ML potential (default checkpoint: uma-s-1p2p1)
  * Environment: `.venv` at the repo root; `requirements.txt` lists fairchem-core + rdkit (ase/torch come with fairchem-core)

## Project Structure
code/          # all code
results/       # any new produced files

## Other information
- the UMA weights will be pulled from HuggingFace; assume that the HuggingFace API key is an environment variable
- the `uma-s-1p2p1` checkpoint is cached in `~/.cache/fairchem` — no download needed