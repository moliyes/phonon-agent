from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .private_config import load_user_config
from .util import safe_name, sha256_file, sha256_json


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except Exception as exc:
        raise ConfigError(f"{name} must be three positive integers") from exc
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ConfigError(f"{name} must be three positive integers")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class WorkflowConfig:
    path: Path
    raw: dict[str, Any]
    name: str
    runs_dir: Path
    structure: Path
    model: str
    model_path: Path | None
    device: str
    deepmd_python: Path | None
    ph_command: str
    scripts: dict[str, Path]
    precheck: dict[str, Any]
    md: dict[str, Any]
    selection: dict[str, Any]
    dft: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def fingerprint(self) -> str:
        payload = dict(self.raw)
        dft = dict(payload.get("dft") or {})
        dispatcher = dict(dft.get("dispatcher") or {})
        # Credential files are references only. Never hash or read their content.
        dispatcher["machine"] = str(self.dft["dispatcher"]["machine"])
        dispatcher["resources"] = str(self.dft["dispatcher"]["resources"])
        dispatcher["cores"] = self.dft["dispatcher"]["cores"]
        dispatcher["command"] = self.dft["dispatcher"]["command"]
        dft["dispatcher"] = dispatcher
        potcar = dict(dft.get("potcar") or {})
        potcar.pop("archive", None)
        potcar["source"] = str(self.dft["potcar"]["source"]) if self.dft["potcar"]["source"] else None
        potcar["family"] = self.dft["potcar"]["family"]
        dft["potcar"] = potcar
        payload["dft"] = dft
        files: dict[str, str] = {"structure": sha256_file(self.structure)}
        for key in ("incar", "kpoints"):
            path = self.dft.get(key)
            if isinstance(path, Path) and path.is_file():
                files[key] = sha256_file(path)
        payload["input_hashes"] = files
        payload["resolved_environment"] = {
            "deepmd_python": str(self.deepmd_python) if self.deepmd_python else None,
            "ph_command": self.ph_command,
        }
        return sha256_json(payload)


def load_config(target: Path) -> WorkflowConfig:
    target = target.expanduser().resolve()
    path = target / "workflow.yaml" if target.is_dir() else target
    if not path.is_file():
        raise ConfigError(f"workflow config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("workflow.yaml top level must be a mapping")
    if int(raw.get("schema_version", 0)) != 2:
        raise ConfigError(
            "schema_version must be 2; recreate this test workspace with "
            "phonon-agent init <case>"
        )
    base = path.parent
    user_config = load_user_config()
    project = raw.get("project") or {}
    name = safe_name(str(project.get("name") or base.name))
    runs_dir = _resolve(base, project.get("runs_dir", "runs"))
    structure_raw = raw.get("structure") or {}
    structure = _resolve(base, structure_raw.get("file", "inputs/POSCAR"))
    model_raw = raw.get("model") or {}
    model = str(model_raw.get("pretrained", "builtin:dpa4"))
    model_path = None if model == "builtin:dpa4" else _resolve(base, model)
    device = str(model_raw.get("device", "cuda:0"))
    if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ConfigError("model.device must be cpu or cuda:N")
    environments = dict(raw.get("environments") or {})
    deepmd_environment = dict(environments.get("deepmd") or {})
    explicit_python = deepmd_environment.get("python")
    deepmd_python = _resolve(base, explicit_python) if explicit_python else user_config.deepmd_python
    tools = dict(raw.get("tools") or {})
    ph_command = str(tools.get("ph_command", user_config.ph_command))
    scripts_raw = raw.get("scripts") or {}
    if not isinstance(scripts_raw, dict):
        raise ConfigError("scripts must be a stage-to-path mapping")
    required_scripts = (
        "phonon-check", "md", "entropy", "selection", "dft-prepare", "dft-submit"
    )
    scripts: dict[str, Path] = {}
    for stage in required_scripts:
        value = scripts_raw.get(stage)
        if not value:
            raise ConfigError(f"scripts.{stage} is required")
        script = _resolve(base, value)
        try:
            script.relative_to(base.resolve())
        except ValueError as exc:
            raise ConfigError(f"scripts.{stage} must stay inside the case directory") from exc
        if script.suffix != ".py":
            raise ConfigError(f"scripts.{stage} must point to a Python file")
        scripts[stage] = script
    collect_value = scripts_raw.get("dft-collect", "scripts/60_collect_dft.py")
    collect_script = _resolve(base, collect_value)
    try:
        collect_script.relative_to(base.resolve())
    except ValueError as exc:
        raise ConfigError("scripts.dft-collect must stay inside the case directory") from exc
    if collect_script.suffix != ".py":
        raise ConfigError("scripts.dft-collect must point to a Python file")
    scripts["dft-collect"] = collect_script

    precheck = dict(raw.get("precheck") or {})
    precheck["supercell"] = _triple(precheck.get("supercell", [2, 2, 2]), "precheck.supercell")
    precheck["mesh"] = _triple(precheck.get("mesh", [20, 20, 20]), "precheck.mesh")
    precheck["imaginary_stop_thz"] = float(precheck.get("imaginary_stop_thz", -0.1))
    precheck["enabled"] = bool(precheck.get("enabled", True))

    md = dict(raw.get("md") or {})
    md["ensemble"] = str(md.get("ensemble", "nvt")).lower()
    if md["ensemble"] not in {"nvt", "npt"}:
        raise ConfigError("md.ensemble must be nvt or npt")
    md["repeat"] = _triple(md.get("repeat", [1, 1, 1]), "md.repeat")
    md["temperatures_k"] = [float(item) for item in md.get("temperatures_k", [50])]
    if not md["temperatures_k"] or any(item <= 0 for item in md["temperatures_k"]):
        raise ConfigError("md.temperatures_k must contain positive values")
    for key, default in (("timestep_fs", 1.0), ("tdamp_fs", 100.0), ("pdamp_fs", 1000.0)):
        md[key] = float(md.get(key, default))
    for key, default in (("equilibration_steps", 2000), ("production_steps", 10000), ("save_interval_steps", 10), ("random_seed", 42)):
        md[key] = int(md.get(key, default))
    md["pressure_gpa"] = float(md.get("pressure_gpa", 0.0))
    md["safety"] = dict(md.get("safety") or {})

    selection = dict(raw.get("selection") or {})
    selection["count"] = None if selection.get("count") in (None, "") else int(selection["count"])
    selection["max_structures"] = int(selection.get("max_structures", 100))
    selection["entropy_fraction"] = float(selection.get("entropy_fraction", 0.95))
    selection["include_initial_supercell"] = bool(selection.get("include_initial_supercell", True))
    selection["k"] = int(selection.get("k", 32))
    selection["cutoff_angstrom"] = float(selection.get("cutoff_angstrom", 5.0))
    selection["bandwidth_inv_angstrom"] = float(selection.get("bandwidth_inv_angstrom", 0.015))
    selection["batch_size"] = int(selection.get("batch_size", 1000))
    selection["pool_stride"] = int(selection.get("pool_stride", 1))
    if not 0 < selection["entropy_fraction"] <= 1:
        raise ConfigError("selection.entropy_fraction must be in (0, 1]")

    dft = dict(raw.get("dft") or {})
    dft["incar"] = _resolve(base, dft.get("incar", "inputs/INCAR"))
    dft["kpoints"] = _resolve(base, dft["kpoints"]) if dft.get("kpoints") else None
    potcar = dict(dft.get("potcar") or {})
    explicit_source = potcar.get("source", potcar.get("archive"))
    potcar["source"] = _resolve(base, explicit_source) if explicit_source else user_config.potcar_source
    # Retain the resolved alias for callers loading an older workflow.
    potcar["archive"] = potcar["source"]
    potcar["family"] = str(
        potcar["family"] if "family" in potcar else user_config.potcar_family
    ).strip("/")
    potcar["variants"] = dict(potcar.get("variants") or {})
    dft["potcar"] = potcar
    dispatcher = dict(dft.get("dispatcher") or {})
    dispatcher["machine"] = (
        _resolve(base, dispatcher["machine"]) if dispatcher.get("machine") else user_config.machine
    )
    dispatcher["resources"] = (
        _resolve(base, dispatcher["resources"]) if dispatcher.get("resources") else user_config.resources
    )
    dispatcher["cores"] = int(dispatcher.get("cores", user_config.cores))
    if dispatcher["cores"] <= 0:
        raise ConfigError("dft.dispatcher.cores must be positive")
    command_template = str(dispatcher.get("command_template", user_config.command_template))
    try:
        default_command = command_template.format(cores=dispatcher["cores"])
    except (KeyError, ValueError) as exc:
        raise ConfigError("dft.dispatcher.command_template may only require {cores}") from exc
    dispatcher["command"] = str(dispatcher.get("command", default_command))
    dispatcher["clean_remote_after_success"] = bool(dispatcher.get("clean_remote_after_success", False))
    dispatcher["user_config"] = str(user_config.path)
    dft["dispatcher"] = dispatcher

    return WorkflowConfig(
        path=path,
        raw=raw,
        name=name,
        runs_dir=runs_dir,
        structure=structure,
        model=model,
        model_path=model_path,
        device=device,
        deepmd_python=deepmd_python,
        ph_command=ph_command,
        scripts=scripts,
        precheck=precheck,
        md=md,
        selection=selection,
        dft=dft,
    )
