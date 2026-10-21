from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .workspace import RunWorkspace


def _task_name(outcar: Path, jobs_dir: Path) -> str:
    try:
        return str(outcar.parent.relative_to(jobs_dir))
    except ValueError:
        return str(outcar.parent)


def collect_dft(workspace: RunWorkspace) -> dict[str, Any]:
    """Convert every dpdata-readable OUTCAR into a DeepMD NPY dataset."""

    jobs_dir = workspace.stage("dft-prepare") / "jobs"
    outcars = sorted(jobs_dir.rglob("OUTCAR")) if jobs_dir.is_dir() else []
    if not outcars:
        raise FileNotFoundError(f"no OUTCAR files found under: {jobs_dir}")

    try:
        import dpdata as dd
    except Exception as exc:
        raise ImportError(f"cannot import dpdata: {type(exc).__name__}") from exc

    subsystems = []
    failed_tasks: list[str] = []
    empty_tasks: list[str] = []
    total_frames = 0
    for index, outcar in enumerate(outcars, start=1):
        task = _task_name(outcar, jobs_dir)
        print(f"[{index}/{len(outcars)}] reading {task}/OUTCAR", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subsystem = dd.LabeledSystem(str(outcar))
        except Exception:
            failed_tasks.append(task)
            continue
        if len(subsystem) == 0:
            empty_tasks.append(task)
            continue
        subsystems.append(subsystem)
        total_frames += len(subsystem)

    if not subsystems:
        raise ValueError(
            "dpdata could not convert any OUTCAR "
            f"(found={len(outcars)}, empty={len(empty_tasks)}, failed={len(failed_tasks)})"
        )

    multi = dd.MultiSystems(*subsystems)
    stage = workspace.stage("dft-collect")
    output = stage / "dpdata"
    output.mkdir(parents=True, exist_ok=False)
    multi.to("deepmd/npy", str(output))
    systems = sorted(str(name) for name in multi.systems)
    result = {
        "status": "completed",
        "jobs_dir": str(jobs_dir),
        "outcar_count": len(outcars),
        "successful_outcars": len(subsystems),
        "empty_outcars": len(empty_tasks),
        "failed_outcars": len(failed_tasks),
        "system_count": len(systems),
        "systems": systems,
        "total_frames": total_frames,
        "dpdata_dir": str(output),
        "empty_tasks": empty_tasks,
        "failed_tasks": failed_tasks,
    }
    (stage / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "dpdata conversion completed: "
        f"{len(subsystems)}/{len(outcars)} OUTCAR, "
        f"{total_frames} frame(s), {len(systems)} system(s)",
        flush=True,
    )
    return result
