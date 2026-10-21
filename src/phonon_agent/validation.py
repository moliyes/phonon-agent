from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any
from zipfile import is_zipfile

from ase.io import read

from .config import WorkflowConfig
from .dft import element_order, parse_incar
from .errors import ConfigError
from .private_config import (
    DEEPMD_ENV_GUIDANCE,
    PRIVATE_CONFIG_GUIDANCE,
    command_core_count,
    load_machine_safely,
    machine_core_count,
    probe_python_module,
)


STAGE_ORDER = (
    "phonon-check", "md", "entropy", "selection",
    "dft-prepare", "dft-submit", "dft-collect",
)


def _includes(until: str, stage: str) -> bool:
    if until not in STAGE_ORDER:
        raise ConfigError(f"unknown validation stage: {until}")
    return STAGE_ORDER.index(until) >= STAGE_ORDER.index(stage)


def _contains_secret(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key.lower() in {"password", "access_key", "app_key", "token", "ticket"}
            or _contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return False


def validate_config(
    config: WorkflowConfig,
    *,
    until: str = "dft-collect",
    check_dispatcher: bool | None = None,
) -> dict[str, Any]:
    # Backward-compatible adapter for the original public Python API.
    if check_dispatcher is False and until == "dft-submit":
        until = "dft-prepare"
    errors: list[str] = []
    warnings: list[str] = []
    for stage in STAGE_ORDER[: STAGE_ORDER.index(until) + 1]:
        script = config.scripts[stage]
        if not script.is_file():
            errors.append(f"stage script missing: {script}")
    if not config.structure.is_file():
        errors.append(f"structure missing: {config.structure}")
        elements = []
    else:
        try:
            atoms = read(config.structure, format="vasp")
            if "REPLACE" in config.structure.read_text(encoding="utf-8", errors="ignore").splitlines()[0].upper():
                errors.append("replace the placeholder POSCAR")
            if len(atoms) == 0 or abs(float(atoms.get_volume())) < 1e-12:
                errors.append("structure must have atoms and a non-singular cell")
            elements = element_order(config.structure)
        except Exception as exc:
            errors.append(f"cannot parse structure: {type(exc).__name__}: {exc}")
            elements = []
    if _includes(until, "phonon-check") and config.model_path is not None and not config.model_path.is_file():
        errors.append(f"model missing: {config.model_path}")
    ph = shutil.which(config.ph_command) if "/" not in config.ph_command else config.ph_command
    if config.precheck["enabled"] and (not ph or not Path(ph).is_file()):
        errors.append(f"phonon-kit CLI missing: {config.ph_command}")
    needs_deepmd = config.precheck["enabled"] or _includes(until, "md")
    if needs_deepmd:
        if config.deepmd_python is None:
            errors.append(DEEPMD_ENV_GUIDANCE)
        else:
            deepmd_ok, _ = probe_python_module(config.deepmd_python, "deepmd")
            if not deepmd_ok:
                errors.append(f"DeepMD-kit is not importable with configured Python: {config.deepmd_python}")
            if _includes(until, "md") and Path(sys.executable).resolve() != config.deepmd_python.resolve():
                errors.append(
                    "phonon-agent is not running in the configured DeepMD environment; "
                    "use ~/.local/bin/phonon-agent"
                )
    required_modules = []
    if _includes(until, "entropy"):
        required_modules.append("quests")
    if _includes(until, "dft-submit"):
        required_modules.append("dpdispatcher")
    if _includes(until, "dft-collect"):
        required_modules.append("dpdata")
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            errors.append(f"Python module missing: {module}")
    if _includes(until, "dft-prepare"):
        incar = config.dft["incar"]
        if not incar.is_file():
            errors.append(f"INCAR missing: {incar}")
            tags = {}
        else:
            tags = parse_incar(incar)
            if tags.get("IBRION") != "-1" or tags.get("NSW") != "0":
                errors.append("INCAR must keep IBRION=-1 and NSW=0 for static labels")
        kpoints = config.dft.get("kpoints")
        if "KSPACING" not in tags:
            if not isinstance(kpoints, Path) or not kpoints.is_file():
                errors.append("INCAR has no KSPACING; dft.kpoints is required")
            elif "REPLACE" in kpoints.read_text(encoding="utf-8", errors="ignore").splitlines()[0].upper():
                errors.append("replace the placeholder KPOINTS")
        potcar = config.dft["potcar"]
        source = potcar.get("source")
        if not isinstance(source, Path) or not source.exists():
            errors.append(PRIVATE_CONFIG_GUIDANCE)
        elif not source.is_dir() and not is_zipfile(source):
            errors.append(f"POTCAR source must be a ZIP archive or directory: {source}")
        missing_variants = [element for element in elements if not potcar["variants"].get(element)]
        if missing_variants:
            errors.append("explicit POTCAR variants missing: " + ", ".join(missing_variants))
    dispatcher = config.dft["dispatcher"]
    if _includes(until, "dft-submit"):
        missing = [key for key in ("machine", "resources") if not dispatcher[key].is_file()]
        if missing:
            errors.append(PRIVATE_CONFIG_GUIDANCE)
        if dispatcher["machine"].is_file():
            try:
                machine = load_machine_safely(dispatcher["machine"])
                if _contains_secret(machine) and stat.S_IMODE(dispatcher["machine"].stat().st_mode) != 0o600:
                    errors.append("Bohrium machine config containing credentials must have permissions 0600")
                profile = machine.get("remote_profile") or {}
                has_legacy = bool((profile.get("email") or profile.get("phone")) and profile.get("password"))
                has_openapi = bool(
                    (profile.get("access_key") or os.environ.get("BOHRIUM_ACCESS_KEY"))
                    and (profile.get("project_id") or os.environ.get("BOHRIUM_PROJECT_ID"))
                )
                if not (has_legacy or has_openapi):
                    errors.append("Bohrium authentication is incomplete; run phonon-agent setup")
                machine_cores = machine_core_count(machine)
                if machine_cores is None:
                    errors.append("cannot determine core count from Bohrium machine type")
                elif machine_cores != dispatcher["cores"]:
                    errors.append(
                        f"Bohrium machine has {machine_cores} cores but dispatcher declares {dispatcher['cores']}"
                    )
            except ConfigError as exc:
                errors.append(str(exc))
        command_cores = command_core_count(dispatcher["command"])
        if command_cores != dispatcher["cores"]:
            errors.append(
                f"VASP command must use mpirun/mpiexec -n {dispatcher['cores']} to match the Bohrium machine"
            )
    if config.selection["max_structures"] < 1:
        errors.append("selection.max_structures must be positive")
    if config.selection["count"] and config.selection["count"] > config.selection["max_structures"]:
        warnings.append("selection.count exceeds max_structures; entropy order will be too short")
    errors = list(dict.fromkeys(errors))
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "elements": elements,
        "configuration": str(config.path),
        "validated_until": until,
        "dispatcher_references_only": True,
    }


def assert_valid(report: dict[str, Any]) -> None:
    if not report["ok"]:
        raise ConfigError("configuration validation failed:\n- " + "\n- ".join(report["errors"]))
