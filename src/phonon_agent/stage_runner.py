from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import yaml

from .config import WorkflowConfig, load_config
from .errors import ConfigError, WaitingForUser
from .state import StateStore
from .util import atomic_write_json, redact, sha256_file, sha256_json, utc_now
from .validation import assert_valid, validate_config
from .workspace import RunWorkspace, create_or_resume, latest_run


EXECUTABLE_STAGES = (
    "phonon-check",
    "md",
    "entropy",
    "selection",
    "dft-prepare",
    "dft-submit",
    "dft-collect",
)
SUCCESS_STATES = {"completed", "skipped"}


@dataclass(frozen=True)
class StageContext:
    """Stable interface passed to an editable workspace stage script."""

    workspace: RunWorkspace
    stage_name: str
    attempt_dir: Path
    script_path: Path
    wait: bool = False

    @property
    def root(self) -> Path:
        return self.workspace.root

    @property
    def config(self) -> WorkflowConfig:
        return self.workspace.config

    @property
    def store(self) -> StateStore:
        return self.workspace.store

    def stage(self, name: str) -> Path:
        if name == self.stage_name:
            return self.attempt_dir
        path = self.workspace.stage(name)
        if path == self.workspace.stage_root(name):
            raise ConfigError(f"stage {name} has no successful current attempt")
        return path


class _Tee:
    def __init__(self, terminal: TextIO, log: TextIO):
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def _file_hash(path: Path | None) -> str | None:
    return sha256_file(path) if isinstance(path, Path) and path.is_file() else None


def stage_input_payload(config: WorkflowConfig, stage: str) -> dict[str, Any]:
    common = {
        "structure": _file_hash(config.structure),
        "model": config.model,
        "model_file": _file_hash(config.model_path),
        "device": config.device,
        "deepmd_python": str(config.deepmd_python) if config.deepmd_python else None,
    }
    if stage == "phonon-check":
        return {**common, "precheck": config.precheck, "ph_command": config.ph_command}
    if stage == "md":
        return {**common, "md": config.md}
    if stage == "entropy":
        selection = {
            key: value
            for key, value in config.selection.items()
            if key != "count"
        }
        return {"selection": selection}
    if stage == "selection":
        return {"selection_count": config.selection["count"]}
    if stage == "dft-prepare":
        potcar = config.dft["potcar"]
        return {
            "incar": _file_hash(config.dft["incar"]),
            "kpoints": _file_hash(config.dft.get("kpoints")),
            "potcar_source": str(potcar.get("source")),
            "potcar_family": potcar.get("family"),
            "potcar_variants": potcar.get("variants"),
        }
    if stage == "dft-submit":
        dispatcher = config.dft["dispatcher"]
        return {
            "machine": str(dispatcher["machine"]),
            "resources": str(dispatcher["resources"]),
            "cores": dispatcher["cores"],
            "command": dispatcher["command"],
            "clean_remote_after_success": dispatcher["clean_remote_after_success"],
        }
    if stage == "dft-collect":
        return {"format": "deepmd/npy", "filters": "none"}
    raise ConfigError(f"unknown stage: {stage}")


def stage_input_fingerprint(config: WorkflowConfig, stage: str) -> str:
    return sha256_json(stage_input_payload(config, stage))


def _check_completed_inputs(config: WorkflowConfig, state: dict[str, Any], stage: str) -> None:
    end = EXECUTABLE_STAGES.index(stage)
    for previous in EXECUTABLE_STAGES[:end]:
        record = state["stages"][previous]
        if record.get("status") not in SUCCESS_STATES:
            raise ConfigError(
                f"prerequisite {previous} is {record.get('status', 'pending')}; "
                f"run: phonon-agent stage {config.base_dir} {previous}"
            )
        recorded = record.get("input_fingerprint")
        current = stage_input_fingerprint(config, previous)
        if recorded != current:
            raise ConfigError(
                f"configuration now changes completed stage {previous}; "
                "start a new run with phonon-check --new"
            )


def _load_script(path: Path, stage: str, attempt: int) -> ModuleType:
    module_name = f"_phonon_agent_{stage.replace('-', '_')}_{attempt}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load stage script: {path}")
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    if not callable(getattr(module, "run", None)):
        raise ConfigError(f"stage script must define run(context): {path}")
    return module


def _publish_attempt(workspace: RunWorkspace, stage: str, attempt_dir: Path) -> None:
    stage_root = workspace.stage_root(stage)
    temporary = stage_root / ".current-new"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(attempt_dir.name, target_is_directory=True)
    os.replace(temporary, stage_root / "current")


def _record_attempt(
    store: StateStore,
    stage: str,
    attempt_record: dict[str, Any],
    status: str,
    **details: Any,
) -> None:
    state = store.load()
    current = dict(state["stages"].get(stage) or {})
    if status == "running":
        for key in ("reason", "error_type", "message"):
            current.pop(key, None)
    attempts = list(current.get("attempts") or [])
    if attempts and attempts[-1].get("number") == attempt_record["number"]:
        attempts[-1] = attempt_record
    else:
        attempts.append(attempt_record)
    current.update(details)
    current["attempts"] = attempts
    current.pop("status", None)
    store.update_stage(stage, status, **current)


def _finish_workflow_status(workspace: RunWorkspace) -> None:
    state = workspace.store.load()
    statuses = [state["stages"][name]["status"] for name in EXECUTABLE_STAGES]
    if state["stages"]["dft-submit"]["status"] == "submitted":
        state["status"] = "waiting_dft"
        state["next_stage"] = "dft-collect"
    elif all(value in SUCCESS_STATES for value in statuses):
        state["status"] = "completed"
        state["next_stage"] = None
    elif not any(value in {"failed", "waiting_user", "running"} for value in statuses):
        state["status"] = "ready"
        state["next_stage"] = next(
            name
            for name in EXECUTABLE_STAGES
            if state["stages"][name]["status"] == "pending"
        )
    workspace.store.save(state)


def _wait_for_submitted(config: WorkflowConfig, workspace: RunWorkspace) -> RunWorkspace:
    return run_stage(
        config,
        "dft-submit",
        wait=True,
        explicit_workspace=workspace.root,
    )


def run_stage(
    config: WorkflowConfig,
    stage: str,
    *,
    force_new: bool = False,
    wait: bool = False,
    explicit_workspace: Path | None = None,
) -> RunWorkspace:
    if stage not in EXECUTABLE_STAGES:
        raise ConfigError(f"unknown stage: {stage}")
    if force_new and stage != "phonon-check":
        raise ConfigError("--new is only valid when starting phonon-check")
    report = validate_config(config, until=stage)
    assert_valid(report)
    if explicit_workspace is not None:
        workspace = RunWorkspace(explicit_workspace.expanduser().resolve(), config)
    else:
        latest = latest_run(config)
        if latest is None and stage != "phonon-check":
            raise ConfigError(
                f"no run exists; start with: phonon-agent stage {config.base_dir} phonon-check"
            )
        workspace = create_or_resume(config, force_new=force_new)

    if sha256_file(config.structure) != sha256_file(workspace.root / "inputs" / "POSCAR"):
        raise ConfigError("the case structure changed after this run started; start phonon-check --new")
    state = workspace.store.load()
    if stage == "dft-collect" and state["stages"]["dft-submit"]["status"] == "submitted":
        _wait_for_submitted(config, workspace)
        state = workspace.store.load()
    _check_completed_inputs(config, state, stage)
    current = state["stages"][stage]
    source_script = config.scripts[stage]
    if not source_script.is_file():
        raise ConfigError(f"stage script is missing: {source_script}")
    source_hash = sha256_file(source_script)
    if current.get("status") in SUCCESS_STATES:
        if current.get("script_sha256") != source_hash:
            raise ConfigError(
                f"{stage} is complete but its workspace script changed; "
                "start a new run to apply the change"
            )
        return workspace
    if current.get("status") == "submitted":
        if current.get("script_sha256") != source_hash:
            raise ConfigError("cannot change the dft-submit script after remote submission")
        if current.get("input_fingerprint") != stage_input_fingerprint(config, stage):
            raise ConfigError("cannot change dispatcher settings after remote submission")
    if current.get("status") == "submitted" and stage != "dft-submit":
        raise ConfigError(f"only dft-submit may have submitted status")

    attempt_number = len(current.get("attempts") or []) + 1
    attempt_dir = workspace.stage_root(stage) / f"attempt-{attempt_number:03d}"
    attempt_dir.mkdir(parents=False)
    config_snapshot = redact(dict(config.raw))
    config_snapshot["resolved_for_stage"] = {
        "stage": stage,
        "input_fingerprint": stage_input_fingerprint(config, stage),
        "structure_snapshot": str(workspace.root / "inputs" / "POSCAR"),
        "deepmd_python": str(config.deepmd_python) if config.deepmd_python else None,
        "phonon_kit_command": config.ph_command,
        "potcar_source_reference": str(config.dft["potcar"].get("source")),
        "dispatcher_machine_reference": str(config.dft["dispatcher"]["machine"]),
        "dispatcher_resources_reference": str(config.dft["dispatcher"]["resources"]),
    }
    (attempt_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    snapshot_dir = workspace.root / "script-snapshots" / stage
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"attempt-{attempt_number:03d}-{source_hash[:12]}.py"
    shutil.copy2(source_script, snapshot)
    snapshot.chmod(0o444)
    log_path = workspace.root / "logs" / f"{stage}-attempt-{attempt_number:03d}.log"
    attempt_record = {
        "number": attempt_number,
        "status": "running",
        "started_at": utc_now(),
        "script": str(snapshot),
        "script_sha256": source_hash,
        "output_dir": str(attempt_dir),
        "log": str(log_path),
    }
    _record_attempt(workspace.store, stage, attempt_record, "running")

    context = StageContext(workspace, stage, attempt_dir, snapshot, wait=wait)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                module = _load_script(snapshot, stage, attempt_number)
                result = module.run(context)
        if not isinstance(result, dict):
            raise ConfigError(f"{stage} script run(context) must return a mapping")
        status = str(result.pop("status", "completed"))
        if status not in {"completed", "skipped", "submitted"}:
            raise ConfigError(f"{stage} returned unsupported status: {status}")
        reserved = {
            "attempts", "input_fingerprint", "script_sha256",
            "script_snapshot", "attempt_dir",
        }
        overlap = sorted(reserved.intersection(result))
        if overlap:
            raise ConfigError(
                f"{stage} result uses reserved keys: {', '.join(overlap)}"
            )
        atomic_write_json(attempt_dir / "result.json", {"status": status, **result})
        _publish_attempt(workspace, stage, attempt_dir)
        attempt_record.update({"status": status, "finished_at": utc_now()})
        _record_attempt(
            workspace.store,
            stage,
            attempt_record,
            status,
            input_fingerprint=stage_input_fingerprint(config, stage),
            script_sha256=source_hash,
            script_snapshot=str(snapshot),
            attempt_dir=str(attempt_dir),
            **result,
        )
    except WaitingForUser as exc:
        attempt_record.update(
            {"status": "waiting_user", "finished_at": utc_now(), "reason": str(exc)}
        )
        _record_attempt(
            workspace.store, stage, attempt_record, "waiting_user", reason=str(exc)
        )
    except Exception as exc:
        attempt_record.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        _record_attempt(
            workspace.store,
            stage,
            attempt_record,
            "failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
    _finish_workflow_status(workspace)
    return workspace


def resume_workspace(run_dir: Path, *, wait: bool = False) -> RunWorkspace:
    run_dir = run_dir.expanduser().resolve()
    state = StateStore(run_dir / "state.json").load()
    config = load_config(Path(state["source_config"]))
    active = [
        name
        for name in EXECUTABLE_STAGES
        if state["stages"][name]["status"] in {"failed", "waiting_user", "running", "submitted"}
    ]
    if not active:
        raise ConfigError("run has no failed, waiting, running, or submitted stage to resume")
    return run_stage(
        config,
        active[0],
        wait=wait,
        explicit_workspace=run_dir,
    )
