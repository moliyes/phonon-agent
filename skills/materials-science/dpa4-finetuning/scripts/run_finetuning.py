#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
INPUT_TEMPLATE = SKILL_ROOT / "assets" / "input.json"
STEP_MULTIPLIER = 30


class FineTuningError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def configured_deepmd_python() -> Path:
    command = shutil.which("phonon-agent")
    if not command:
        raise FineTuningError(
            "phonon-agent command not found; run phonon-agent setup first"
        )
    result = subprocess.run(
        [command, "setup", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FineTuningError(
            "cannot read the global phonon-agent configuration; "
            "run phonon-agent setup"
        ) from exc
    value = report.get("deepmd_python")
    if not value:
        raise FineTuningError(
            "DeepMD Python is not configured; run phonon-agent setup"
        )
    python = Path(str(value)).expanduser().resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FineTuningError(
            f"configured DeepMD Python is not executable: {python}; "
            "run phonon-agent setup"
        )
    return python


def enter_configured_environment(python: Path, argv: list[str]) -> None:
    if Path(sys.executable).resolve() != python:
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *argv])


def resolve_dp_command(python: Path) -> Path:
    command = python.parent / "dp"
    if not command.is_file() or not os.access(command, os.X_OK):
        raise FineTuningError(
            f"DeepMD dp command is missing beside the configured Python: {command}"
        )
    return command


def default_model_path() -> Path:
    try:
        from phonon_kit.model_manager import packaged_checkpoint_path
    except Exception as exc:
        raise FineTuningError(
            "phonon-kit is unavailable in the configured DeepMD environment; "
            "run phonon-agent setup"
        ) from exc
    return packaged_checkpoint_path().resolve()


def validate_model(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".pt":
        raise FineTuningError(
            f"fine-tuning requires a DPA4 .pt checkpoint, not {path.suffix or 'a suffixless file'}"
        )
    if not path.is_file():
        raise FineTuningError(f"DPA4 checkpoint does not exist: {path}")
    return path


def inspect_dataset(data: Path) -> tuple[list[str], int]:
    try:
        import numpy as np
    except Exception as exc:
        raise FineTuningError(
            "NumPy is unavailable in the configured DeepMD environment"
        ) from exc

    data = data.expanduser().resolve()
    if not data.is_dir():
        raise FineTuningError(f"DeepMD data directory does not exist: {data}")
    systems = sorted({path.parent for path in data.rglob("type.raw")})
    if not systems:
        raise FineTuningError(
            f"no DeepMD NPY systems containing type.raw were found under: {data}"
        )

    names: list[str] = []
    total_frames = 0
    for system in systems:
        relative = system.relative_to(data)
        names.append(data.name if str(relative) == "." else relative.as_posix())
        coordinates = sorted(system.glob("set.*/coord.npy"))
        if not coordinates:
            raise FineTuningError(f"system has no set.*/coord.npy: {system}")
        for coordinate in coordinates:
            try:
                array = np.load(coordinate, mmap_mode="r")
            except Exception as exc:
                raise FineTuningError(
                    f"cannot read DeepMD coordinates: {coordinate}"
                ) from exc
            if array.ndim < 1 or int(array.shape[0]) <= 0:
                raise FineTuningError(f"DeepMD coordinate set is empty: {coordinate}")
            total_frames += int(array.shape[0])
    return names, total_frames


def _commands(dp_command: Path) -> list[dict[str, Any]]:
    return [
        {
            "step": "train",
            "argv": [
                str(dp_command),
                "--pt",
                "train",
                "input.json",
                "--finetune",
                "dpa4_pretrained.pt",
            ],
            "log": "train_log",
        },
        {
            "step": "freeze",
            "argv": [
                str(dp_command),
                "--pt",
                "freeze",
                "-o",
                "frozen.pt2",
            ],
            "log": "freeze_log",
        },
        {
            "step": "test",
            "argv": [
                str(dp_command),
                "--pt",
                "test",
                "-m",
                "frozen.pt2",
                "-s",
                "./dpdata",
                "-d",
                "pred_train",
            ],
            "log": "infer_train_log",
        },
    ]


def _write_run_script(output: Path, commands: list[dict[str, Any]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "$0")"']
    for command in commands:
        lines.append(
            f"{shlex.join(command['argv'])} > {shlex.quote(command['log'])} 2>&1"
        )
    path = output / "run_local.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def prepare_workspace(
    data: Path,
    output: Path,
    model: Path,
    dp_command: Path,
    *,
    numb_steps: int | None = None,
    template: Path = INPUT_TEMPLATE,
) -> dict[str, Any]:
    data = data.expanduser().resolve()
    output = output.expanduser().resolve()
    model = validate_model(model)
    dp_command = dp_command.expanduser().resolve()
    if output.exists():
        raise FineTuningError(f"output directory already exists: {output}")
    if output == data or output.is_relative_to(data):
        raise FineTuningError("output directory must not be inside the source data")
    if not dp_command.is_file() or not os.access(dp_command, os.X_OK):
        raise FineTuningError(f"dp command is not executable: {dp_command}")
    if not template.is_file():
        raise FineTuningError(f"input template is missing: {template}")

    systems, total_frames = inspect_dataset(data)
    if numb_steps is not None and numb_steps <= 0:
        raise FineTuningError("--numb-steps must be a positive integer")
    steps = numb_steps if numb_steps is not None else total_frames * STEP_MULTIPLIER

    config = json.loads(template.read_text(encoding="utf-8"))
    if config["training"]["training_data"]["systems"] != "./dpdata":
        raise FineTuningError("input template must use ./dpdata")
    config["training"]["numb_steps"] = steps

    output.mkdir(parents=True)
    shutil.copytree(data, output / "dpdata")
    shutil.copy2(model, output / "dpa4_pretrained.pt")
    (output / "input.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    commands = _commands(dp_command)
    _write_run_script(output, commands)
    summary: dict[str, Any] = {
        "status": "prepared",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "source_data": str(data),
        "source_model": str(model),
        "output_dir": str(output),
        "model_sha256": _sha256(output / "dpa4_pretrained.pt"),
        "systems": systems,
        "system_count": len(systems),
        "total_frames": total_frames,
        "numb_steps": steps,
        "step_rule": "explicit" if numb_steps is not None else "total_frames_x_30",
        "dp_command": str(dp_command),
        "commands": commands,
        "artifacts": {
            "input": str(output / "input.json"),
            "training_curve": str(output / "lcurve.out"),
            "frozen_model": str(output / "frozen.pt2"),
            "prediction_prefix": str(output / "pred_train"),
        },
    }
    _atomic_json(output / "run-summary.json", summary)
    return summary


def _log_tail(path: Path, line_count: int = 20) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])


def execute_workspace(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    summary_path = output / "run-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"status": "running", "started_at": _utc_now(), "updated_at": _utc_now()})
    _atomic_json(summary_path, summary)

    for command in summary["commands"]:
        step = str(command["step"])
        log_path = output / str(command["log"])
        summary.update({"current_step": step, "updated_at": _utc_now()})
        _atomic_json(summary_path, summary)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(value) for value in command["argv"]],
                cwd=output,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            summary.update(
                {
                    "status": "failed",
                    "failed_step": step,
                    "returncode": result.returncode,
                    "finished_at": _utc_now(),
                    "updated_at": _utc_now(),
                }
            )
            summary.pop("current_step", None)
            _atomic_json(summary_path, summary)
            tail = _log_tail(log_path)
            raise FineTuningError(
                f"{step} failed with exit code {result.returncode}; log: {log_path}"
                + (f"\n{tail}" if tail else "")
            )

    frozen = output / "frozen.pt2"
    if not frozen.is_file() or frozen.stat().st_size == 0:
        summary.update(
            {
                "status": "failed",
                "failed_step": "freeze-output",
                "finished_at": _utc_now(),
                "updated_at": _utc_now(),
            }
        )
        summary.pop("current_step", None)
        _atomic_json(summary_path, summary)
        raise FineTuningError("training commands completed but frozen.pt2 is missing or empty")

    summary.update(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "updated_at": _utc_now(),
        }
    )
    summary.pop("current_step", None)
    _atomic_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and run a self-contained local DPA4 fine-tuning workspace."
    )
    parser.add_argument("--data", required=True, type=Path, help="DeepMD NPY data root")
    parser.add_argument("--output", required=True, type=Path, help="new workspace path")
    parser.add_argument("--model", type=Path, help="DPA4 .pt checkpoint")
    parser.add_argument("--numb-steps", type=int, help="override total_frames * 30")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create the workspace without starting DeepMD",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    try:
        python = configured_deepmd_python()
        enter_configured_environment(python, raw_argv)
        dp_command = resolve_dp_command(python)
        model = args.model if args.model is not None else default_model_path()
        summary = prepare_workspace(
            args.data,
            args.output,
            model,
            dp_command,
            numb_steps=args.numb_steps,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.prepare_only:
            return 0
        result = execute_workspace(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (FineTuningError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
