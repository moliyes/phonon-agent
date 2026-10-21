from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import WorkflowConfig
from .errors import ConfigError
from .state import StateStore, initial_state
from .util import redact


STAGE_DIRS = {
    "phonon-check": "00-phonon-check",
    "md": "10-md",
    "entropy": "20-entropy",
    "selection": "30-selection",
    "dft-prepare": "40-dft-prepare",
    "dft-submit": "50-dft-submit",
    "dft-collect": "60-dft-collect",
}


@dataclass(frozen=True)
class RunWorkspace:
    root: Path
    config: WorkflowConfig

    @property
    def store(self) -> StateStore:
        return StateStore(self.root / "state.json")

    def stage(self, name: str) -> Path:
        stage_root = self.stage_root(name)
        current = stage_root / "current"
        return current if current.exists() else stage_root

    def stage_root(self, name: str) -> Path:
        return self.root / "stages" / STAGE_DIRS[name]


def _run_number(path: Path, name: str) -> int:
    values = []
    for item in path.glob(f"{name}-[0-9][0-9][0-9]"):
        try:
            values.append(int(item.name.rsplit("-", 1)[1]))
        except ValueError:
            pass
    return max(values, default=0) + 1


def latest_run(config: WorkflowConfig) -> Path | None:
    candidates = sorted(config.runs_dir.glob(f"{config.name}-[0-9][0-9][0-9]"))
    return candidates[-1] if candidates else None


def create_or_resume(config: WorkflowConfig, *, force_new: bool = False) -> RunWorkspace:
    fingerprint = config.fingerprint()
    latest = latest_run(config)
    if latest and not force_new:
        state = StateStore(latest / "state.json").load()
        if state.get("status") != "completed":
            return RunWorkspace(latest, config)
        raise ConfigError(
            f"latest run is already complete: {latest}; start phonon-check with --new"
        )
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    number = _run_number(config.runs_dir, config.name)
    root = config.runs_dir / f"{config.name}-{number:03d}"
    (root / "inputs").mkdir(parents=True)
    (root / "logs").mkdir()
    for directory in STAGE_DIRS.values():
        (root / "stages" / directory).mkdir(parents=True)
    (root / "script-snapshots").mkdir()
    shutil.copy2(config.structure, root / "inputs" / "POSCAR")
    shutil.copy2(config.dft["incar"], root / "inputs" / "INCAR")
    if config.dft.get("kpoints") and config.dft["kpoints"].is_file():
        shutil.copy2(config.dft["kpoints"], root / "inputs" / "KPOINTS")
    resolved = redact(dict(config.raw))
    resolved["resolved_paths"] = {
        "source_config": str(config.path),
        "structure_snapshot": str(root / "inputs" / "POSCAR"),
        "model": config.model,
        "deepmd_python": str(config.deepmd_python) if config.deepmd_python else None,
        "phonon_kit_command": config.ph_command,
        "potcar_source_reference": (
            str(config.dft["potcar"]["source"]) if config.dft["potcar"]["source"] else None
        ),
        "potcar_family": config.dft["potcar"]["family"],
        "dispatcher_machine_reference": str(config.dft["dispatcher"]["machine"]),
        "dispatcher_resources_reference": str(config.dft["dispatcher"]["resources"]),
        "dispatcher_cores": config.dft["dispatcher"]["cores"],
        "dispatcher_command": config.dft["dispatcher"]["command"],
    }
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    StateStore(root / "state.json").save(initial_state(config.name, fingerprint, config.path))
    return RunWorkspace(root, config)
