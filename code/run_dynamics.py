#!/usr/bin/env python3
"""
Run single-point, geometry-optimization or MD calculations using ASE with
the Fairchem UMA calculator.

This script loads an XYZ file, sets up the UMA ML potential from Fairchem,
and runs one of three calculation modes:
  sp   - single-point energy/force calculation
  opt  - geometry optimization
  md   - molecular dynamics (trajectory XYZ output; forces optional)
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet

# Fairchem calculator for UMA potential
from fairchem.core import FAIRChemCalculator, pretrained_mlip


DEFAULT_CHECKPOINT = "uma-s-1p2p1"


def setup_calculator(atoms, checkpoint: str):
    """Attach the UMA calculator to an Atoms object and return it."""
    # Set charge and spin for the UMA calculator (avoids warnings and
    # potential errors)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1

    # Uses HuggingFace API token from environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = pretrained_mlip.get_predict_unit(checkpoint, device=device)
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    return atoms


def write_frames(frames, output_xyz: str, include_forces: bool) -> None:
    """Write Atoms frames to an extended XYZ file.

    With include_forces=True forces are written as fx/fy/fz columns; with
    False the file is positions-only.
    """
    output = Path(output_xyz)
    output.parent.mkdir(parents=True, exist_ok=True)

    clean = []
    for frame in frames:
        if "momenta" in frame.arrays:
            del frame.arrays["momenta"]
        if not include_forces and "forces" in frame.arrays:
            del frame.arrays["forces"]
        clean.append(frame)

    write(
        output_xyz,
        clean,
        format="extxyz",
        write_results=include_forces,
    )


def run_single_point(
    input_xyz: str,
    include_forces: bool = True,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> None:
    """Compute a single-point energy (and forces) for the input structure."""
    atoms = read(input_xyz)
    print(f"Loaded structure with {len(atoms)} atoms")
    setup_calculator(atoms, checkpoint)
    energy = atoms.get_potential_energy()
    print(f"Single-point energy: {energy:.6f} eV")
    if include_forces:
        forces = atoms.get_forces()
        print("Forces (eV/A):")
        for symbol, f in zip(atoms.get_chemical_symbols(), forces):
            print(f"  {symbol:2s}  {f[0]:+12.6f}  {f[1]:+12.6f}  {f[2]:+12.6f}")


def run_optimization(
    input_xyz: str,
    output_xyz: str,
    fmax: float = 0.01,
    max_steps: int = 500,
    optimizer_name: str = "bfgs",
    include_forces: bool = True,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> None:
    """Optimize the geometry of the input structure with UMA."""
    from ase.optimize import BFGS, FIRE

    optimizers = {"bfgs": BFGS, "fire": FIRE}

    atoms = read(input_xyz)
    print(f"Loaded structure with {len(atoms)} atoms")
    setup_calculator(atoms, checkpoint)

    print(f"Initial energy: {atoms.get_potential_energy():.6f} eV")

    optimizer_cls = optimizers.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    dyn = optimizer_cls(atoms, logfile=None)
    converged = dyn.run(fmax=fmax, steps=max_steps)
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
    print(f"Converged to fmax < {fmax} eV/A: {'yes' if converged else 'NO (max steps reached)'}")
    print(f"Optimization steps: {dyn.nsteps}")

    frame = atoms.copy()
    if include_forces:
        frame.arrays["forces"] = atoms.get_forces()
    write_frames([frame], output_xyz, include_forces)
    print(f"Optimized structure written to: {output_xyz}")


def run_dynamics(
    input_xyz: str,
    output_xyz: str,
    steps: int = 1000,
    timestep: float = 1.0,
    temperature: float = 300.0,
    md_type: str = "langevin",
    include_forces: bool = True,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> None:
    """
    Run molecular dynamics on a structure using UMA ML potential.

    Args:
        input_xyz: Path to input XYZ file.
        output_xyz: Path to write output trajectory XYZ file.
        steps: Number of MD steps to run.
        timestep: Timestep in femtoseconds.
        temperature: Temperature in Kelvin.
        md_type: Type of MD ('verlet' or 'langevin').
        include_forces: Write forces into the trajectory file.
    """
    # Load the structure
    atoms = read(input_xyz)
    print(f"Loaded structure with {len(atoms)} atoms")
    setup_calculator(atoms, checkpoint)

    # Verify calculator works by computing initial energy
    print(f"Initial energy: {atoms.get_potential_energy():.4f} eV")

    # Set up MD
    timestep_seconds = timestep * units.fs

    if md_type == "langevin":
        # Langevin dynamics for NVT ensemble
        dyn = Langevin(
            atoms,
            timestep=timestep_seconds,
            temperature_K=temperature,
            friction=0.02,  # 1/fs, typical for MD
        )
    elif md_type == "verlet":
        # Velocity Verlet for NVE ensemble
        dyn = VelocityVerlet(atoms, timestep=timestep_seconds)
        # Set initial velocities if needed
        ke = atoms.get_kinetic_energy()
        if np.allclose(ke, 0):
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    else:
        raise ValueError(f"Unknown MD type: {md_type}")

    # Create trajectory storage
    trajectory = []

    def record_step():
        """Callback to save each step."""
        frame = atoms.copy()
        if include_forces:
            # Store forces in arrays (not info) to avoid comparison issues
            frame.arrays["forces"] = atoms.get_forces()
        trajectory.append(frame)

    # Attach callback
    dyn.attach(record_step, interval=1)

    # Run MD
    print(f"Running {md_type} MD for {steps} steps at {temperature}K")
    print(f"Timestep: {timestep} fs")
    dyn.run(steps)

    # Write trajectory to XYZ (extended format; forces optional)
    write_frames(trajectory, output_xyz, include_forces)

    print(f"Trajectory written to: {output_xyz}")
    print(f"Total frames: {len(trajectory)}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-point, optimization and MD with the UMA potential (ASE + Fairchem)"
    )
    parser.add_argument(
        "input_xyz",
        help="Input XYZ file",
    )
    parser.add_argument(
        "--mode",
        choices=["sp", "opt", "md"],
        default="md",
        help="Calculation mode: single point, geometry optimization, or MD "
        "(default: md)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/trajectory.xyz",
        help="Output trajectory XYZ file (default: results/trajectory.xyz)",
    )
    parser.add_argument(
        "--no-forces",
        action="store_true",
        help="Do not write forces/gradients to the output file",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="UMA checkpoint name (default: uma-s-1p2p1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of MD steps (default: 1000)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Timestep in femtoseconds (default: 1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    parser.add_argument(
        "--md-type",
        choices=["verlet", "langevin"],
        default="langevin",
        help="MD integrator type (default: langevin)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Force convergence criterion for opt (default: 0.01 eV/A)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum optimizer steps for opt (default: 500)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["bfgs", "fire"],
        default="bfgs",
        help="Optimizer for opt (default: bfgs)",
    )

    args = parser.parse_args()

    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        print(
            "Warning: HF_TOKEN environment variable not set. "
            "Fairchem may fail to download UMA weights.",
            file=sys.stderr,
        )

    try:
        if args.mode == "sp":
            run_single_point(args.input_xyz, not args.no_forces)
        elif args.mode == "opt":
            run_optimization(
                args.input_xyz,
                args.output,
                fmax=args.fmax,
                max_steps=args.max_steps,
                optimizer_name=args.optimizer,
                include_forces=not args.no_forces,
                checkpoint=args.checkpoint,
            )
        else:
            run_dynamics(
                args.input_xyz,
                args.output,
                args.steps,
                args.timestep,
                args.temperature,
                args.md_type,
                not args.no_forces,
                args.checkpoint,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()