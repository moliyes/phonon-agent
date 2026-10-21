from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .errors import ExternalProgramError
from .util import utc_now
from .workspace import RunWorkspace


def temperature_tag(value: float) -> str:
    return f"{int(value):04d}K" if float(value).is_integer() else f"{value:g}K".replace(".", "p")


def _model_path(spec: str) -> Path:
    if spec == "builtin:dpa4":
        try:
            from phonon_kit.config import builtin_model_path
        except Exception as exc:
            raise ExternalProgramError(f"cannot resolve phonon-kit builtin DPA4: {exc}") from exc
        return Path(builtin_model_path())
    return Path(spec)


def _device(value: str) -> None:
    if value == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["DEVICE"] = "cpu"
        os.environ.pop("LOCAL_RANK", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = value.split(":", 1)[1]
        os.environ.pop("DEVICE", None)
        os.environ["LOCAL_RANK"] = "0"


def _minimum_distance(atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _diagnostics(atoms, reference, reference_volume: float) -> dict[str, float]:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    scaled = atoms.get_scaled_positions(wrap=False) - reference.get_scaled_positions(wrap=False)
    scaled -= np.rint(scaled)
    displacement = np.linalg.norm(scaled @ reference.cell.array, axis=1)
    volume = float(atoms.get_volume())
    values = np.concatenate([forces.ravel(), atoms.positions.ravel(), atoms.cell.array.ravel()])
    return {
        "finite": float(bool(np.all(np.isfinite(values)))),
        "temperature_k": float(atoms.get_temperature()),
        "energy_ev_atom": float(atoms.get_potential_energy()) / len(atoms),
        "volume_ratio": volume / reference_volume,
        "max_force_ev_angstrom": float(np.max(np.linalg.norm(forces, axis=1))),
        "min_distance_angstrom": _minimum_distance(atoms),
        "rms_displacement_angstrom": float(np.sqrt(np.mean(displacement**2))),
        "max_displacement_angstrom": float(np.max(displacement)),
    }


def _check(values: dict[str, float], safety: dict[str, Any], ensemble: str) -> None:
    limits = {
        "min_distance_angstrom": (">=", float(safety.get("min_distance_angstrom", 1.0))),
        "max_force_ev_angstrom": ("<=", float(safety.get("max_force_ev_angstrom", 30.0))),
        "rms_displacement_angstrom": ("<=", float(safety.get("max_rms_displacement_angstrom", 1.0))),
        "max_displacement_angstrom": ("<=", float(safety.get("max_atom_displacement_angstrom", 2.0))),
    }
    if ensemble == "npt":
        limits["volume_ratio_min"] = (">=", float(safety.get("min_volume_ratio", 0.75)))
        limits["volume_ratio_max"] = ("<=", float(safety.get("max_volume_ratio", 1.25)))
    failures = []
    if not values["finite"]:
        failures.append("NaN/Inf detected")
    for name, (operator, threshold) in limits.items():
        source = "volume_ratio" if name.startswith("volume_ratio") else name
        actual = values[source]
        failed = actual < threshold if operator == ">=" else actual > threshold
        if failed:
            failures.append(f"{source}={actual:.6g} violates {operator} {threshold:.6g}")
    if failures:
        raise RuntimeError("MD safety check failed: " + "; ".join(failures))


def run_md(workspace: RunWorkspace) -> dict:
    config = workspace.config
    stage = workspace.stage("md")
    if config.deepmd_python is None:
        raise ExternalProgramError("DeepMD environment is not configured; run phonon-agent setup")
    if Path(sys.executable).resolve() != config.deepmd_python.resolve():
        raise ExternalProgramError(
            "phonon-agent is not running in the configured DeepMD environment; "
            "use ~/.local/bin/phonon-agent"
        )
    _device(config.device)
    try:
        from deepmd.calculator import DP
    except Exception as exc:
        raise ExternalProgramError(f"cannot import DeepMD calculator: {exc}") from exc
    unitcell = read(workspace.root / "inputs" / "POSCAR", format="vasp")
    unitcell.pbc = True
    reference = unitcell.repeat(config.md["repeat"])
    write(stage / "supercell.POSCAR", reference, format="vasp", direct=True, vasp5=True, sort=False)
    calculator = DP(model=str(_model_path(config.model)))
    completed = []
    for index, temperature in enumerate(config.md["temperatures_k"]):
        tag = temperature_tag(temperature)
        final_trajectory = stage / f"md_{tag}.traj"
        final_poscar = stage / f"final_{tag}.POSCAR"
        if final_trajectory.is_file() and final_poscar.is_file():
            completed.append(tag)
            continue
        attempt = stage / f"attempt-{tag}-{utc_now().replace(':', '')}"
        attempt.mkdir()
        atoms = reference.copy()
        atoms.calc = calculator
        rng = np.random.default_rng(int(config.md["random_seed"]) + index)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng, force_temp=True)
        Stationary(atoms)
        if config.md["ensemble"] == "npt":
            dynamics = IsotropicMTKNPT(
                atoms,
                timestep=config.md["timestep_fs"] * units.fs,
                temperature_K=temperature,
                pressure_au=config.md["pressure_gpa"] * units.GPa,
                tdamp=config.md["tdamp_fs"] * units.fs,
                pdamp=config.md["pdamp_fs"] * units.fs,
            )
        else:
            dynamics = NoseHooverChainNVT(
                atoms,
                timestep=config.md["timestep_fs"] * units.fs,
                temperature_K=temperature,
                tdamp=config.md["tdamp_fs"] * units.fs,
            )
        reference_volume = float(reference.get_volume())
        interval = int(config.md["save_interval_steps"])
        log_path = attempt / "md.log"
        with log_path.open("w", encoding="utf-8") as log:
            def monitor() -> None:
                values = _diagnostics(atoms, reference, reference_volume)
                _check(values, config.md["safety"], config.md["ensemble"])
                log.write(json.dumps({"step": dynamics.nsteps, **values}) + "\n")
                log.flush()

            dynamics.attach(monitor, interval=interval)
            monitor()
            dynamics.run(int(config.md["equilibration_steps"]))
            with Trajectory(str(attempt / "production.traj"), "w", atoms) as trajectory:
                trajectory.write()
                dynamics.attach(trajectory.write, interval=interval)
                dynamics.run(int(config.md["production_steps"]))
        write(attempt / "final.POSCAR", atoms, format="vasp", direct=True, vasp5=True, sort=False)
        os.replace(attempt / "production.traj", final_trajectory)
        os.replace(attempt / "final.POSCAR", final_poscar)
        os.replace(log_path, stage / f"md_{tag}.log")
        completed.append(tag)
    return {
        "status": "completed",
        "ensemble": config.md["ensemble"],
        "temperatures": completed,
        "atoms_per_frame": len(reference),
        "stage_dir": str(stage),
    }
