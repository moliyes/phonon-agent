from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import yaml

from .errors import ExternalProgramError, WaitingForUser
from .workspace import RunWorkspace


def _command(value: str) -> str:
    resolved = shutil.which(value) if "/" not in value else value
    if not resolved or not Path(resolved).is_file():
        raise ExternalProgramError(
            f"phonon-kit CLI not found: {value}; install phonon-kit in the active environment"
        )
    return str(resolved)


def run_precheck(workspace: RunWorkspace) -> dict:
    config = workspace.config
    stage = workspace.stage("phonon-check")
    if not config.precheck["enabled"]:
        return {"status": "skipped", "reason": "disabled by configuration"}
    summaries = sorted(stage.glob("ph-runs/*/results/pretrained/summary.json"))
    if not summaries:
        case = stage / "case"
        case.mkdir(exist_ok=True)
        structure = workspace.root / "inputs" / "POSCAR"
        payload = {
            "schema_version": 1,
            "project": {"name": "precheck", "runs_dir": str(stage / "ph-runs")},
            "structure": {"file": str(structure)},
            "relaxation": {"enabled": False},
            "phonon": {
                "supercell": list(config.precheck["supercell"]),
                "displacement_angstrom": float(config.precheck.get("displacement_angstrom", 0.01)),
                "primitive": config.precheck.get("primitive", "auto"),
                "symmetry_tolerance": float(config.precheck.get("symmetry_tolerance", 1.0e-5)),
                "band": {"path": "auto", "points_per_segment": 101},
                "mesh": list(config.precheck["mesh"]),
                "thermal": {
                    "temperature_min_k": 0,
                    "temperature_max_k": 300,
                    "temperature_step_k": 100,
                    "imaginary_policy": "exclude",
                    "significant_imaginary_thz": float(config.precheck["imaginary_stop_thz"]),
                },
            },
            "methods": {
                "pretrained": {
                    "type": "deepmd",
                    "model": config.model,
                    "device": config.device,
                }
            },
        }
        ph_config = case / "config.yaml"
        ph_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        result = subprocess.run(
            [_command(config.ph_command), "run", str(ph_config), "--only", "pretrained"],
            cwd=case,
            text=True,
            capture_output=True,
        )
        (stage / "phonon-kit.stdout.log").write_text(result.stdout, encoding="utf-8")
        (stage / "phonon-kit.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise ExternalProgramError(
                f"phonon-kit precheck failed with exit code {result.returncode}; see {stage}"
            )
        summaries = sorted(stage.glob("ph-runs/*/results/pretrained/summary.json"))
    if not summaries:
        raise ExternalProgramError("phonon-kit completed without a pretrained summary.json")
    summary_path = summaries[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    minimum = float(summary["min_frequency_thz"])
    result = {
        "status": "completed",
        "summary": str(summary_path),
        "min_frequency_thz": minimum,
        "stop_threshold_thz": float(config.precheck["imaginary_stop_thz"]),
    }
    if minimum < float(config.precheck["imaginary_stop_thz"]):
        raise WaitingForUser(
            f"pretrained phonon minimum is {minimum:.6g} THz, below "
            f"{config.precheck['imaginary_stop_thz']:.6g} THz; choose relaxation or another structure"
        )
    return result
