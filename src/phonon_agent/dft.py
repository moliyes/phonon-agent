from __future__ import annotations

import json
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any

from ase.io import read

from .errors import ExternalProgramError, WaitingForUser
from .private_config import (
    PRIVATE_CONFIG_GUIDANCE,
    command_core_count,
    dynamic_job_name,
    load_machine_safely,
    machine_core_count,
)
from .workspace import RunWorkspace


BACKWARD_FILES = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "vasp.stdout", "vasp.stderr"]


def parse_incar(path: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("!", 1)[0].split("#", 1)[0]
        for item in line.split(";"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            tags[key.strip().upper()] = value.strip()
    return tags


def element_order(path: Path) -> list[str]:
    atoms = read(path, format="vasp")
    return list(dict.fromkeys(atoms.get_chemical_symbols()))


def validate_dft_inputs(workspace: RunWorkspace, *, require_dispatcher: bool) -> dict[str, Any]:
    config = workspace.config
    incar = config.dft["incar"]
    if not incar.is_file():
        raise WaitingForUser(f"INCAR is missing: {incar}")
    tags = parse_incar(incar)
    conflicts = []
    if tags.get("IBRION") != "-1":
        conflicts.append("IBRION must be -1")
    if tags.get("NSW") != "0":
        conflicts.append("NSW must be 0")
    if conflicts:
        raise WaitingForUser("INCAR is not a static labeling setup: " + "; ".join(conflicts))
    kpoints = config.dft.get("kpoints")
    if "KSPACING" not in tags and (not isinstance(kpoints, Path) or not kpoints.is_file()):
        raise WaitingForUser("INCAR has no KSPACING; provide dft.kpoints")
    if isinstance(kpoints, Path) and kpoints.is_file():
        first_line = kpoints.read_text(encoding="utf-8", errors="ignore").splitlines()[0].upper()
        if "REPLACE" in first_line:
            raise WaitingForUser("replace the placeholder KPOINTS with a user-confirmed mesh")
    potcar = config.dft["potcar"]
    source = potcar.get("source")
    if not isinstance(source, Path) or not source.exists():
        raise WaitingForUser(PRIVATE_CONFIG_GUIDANCE)
    if not source.is_dir() and not zipfile.is_zipfile(source):
        raise WaitingForUser(f"POTCAR source must be a ZIP archive or directory: {source}")
    elements = element_order(workspace.root / "inputs" / "POSCAR")
    missing = [element for element in elements if not potcar["variants"].get(element)]
    if missing:
        raise WaitingForUser(
            "explicit POTCAR variants are required for every task; missing: " + ", ".join(missing)
        )
    if require_dispatcher:
        dispatcher = config.dft["dispatcher"]
        missing_files = [str(dispatcher[key]) for key in ("machine", "resources") if not dispatcher[key].is_file()]
        if missing_files:
            raise WaitingForUser(PRIVATE_CONFIG_GUIDANCE)
        machine = load_machine_safely(dispatcher["machine"])
        profile = machine.get("remote_profile") or {}
        has_legacy = bool((profile.get("email") or profile.get("phone")) and profile.get("password"))
        has_openapi = bool(
            (profile.get("access_key") or os.environ.get("BOHRIUM_ACCESS_KEY"))
            and (profile.get("project_id") or os.environ.get("BOHRIUM_PROJECT_ID"))
        )
        if not (has_legacy or has_openapi):
            raise WaitingForUser("Bohrium authentication is incomplete; run phonon-agent setup")
        actual_cores = machine_core_count(machine)
        requested_cores = int(dispatcher["cores"])
        if actual_cores is None:
            raise WaitingForUser("cannot determine core count from Bohrium machine type")
        if actual_cores != requested_cores:
            raise WaitingForUser(
                f"Bohrium machine has {actual_cores} cores but dispatcher declares {requested_cores}"
            )
        command_cores = command_core_count(dispatcher["command"])
        if command_cores != requested_cores:
            raise WaitingForUser(
                f"VASP command must use mpirun/mpiexec -n {requested_cores} to match the Bohrium machine"
            )
    return {"elements": elements, "incar_tags": tags, "uses_kspacing": "KSPACING" in tags}


def assemble_potcar(workspace: RunWorkspace, destination: Path, elements: list[str]) -> list[str]:
    potcar = workspace.config.dft["potcar"]
    titles: list[str] = []
    source = potcar["source"]
    family = str(potcar["family"]).strip("/")
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        output = destination.open("wb")
        with output:
            for element in elements:
                variant = str(potcar["variants"][element])
                relative = Path(family) / variant / "POTCAR" if family else Path(variant) / "POTCAR"
                member = relative.as_posix()
                try:
                    data = archive.read(member) if archive is not None else (source / relative).read_bytes()
                except (KeyError, FileNotFoundError) as exc:
                    raise WaitingForUser(f"POTCAR variant not found in source: {member}") from exc
                text = data.decode("latin-1", errors="ignore")
                title_match = re.search(r"^\s*TITEL\s*=\s*(.+)$", text, re.MULTILINE)
                if not title_match:
                    raise ValueError(f"POTCAR has no TITEL: {member}")
                title = title_match.group(1).strip()
                if not re.search(rf"(?:^|\s){re.escape(element)}(?:[_\s]|$)", title):
                    raise ValueError(f"POTCAR element mismatch for {element}: {title}")
                titles.append(title)
                output.write(data)
                if not data.endswith(b"\n"):
                    output.write(b"\n")
    finally:
        if archive is not None:
            archive.close()
    return titles


def prepare_dft(workspace: RunWorkspace) -> dict[str, Any]:
    validation = validate_dft_inputs(workspace, require_dispatcher=False)
    stage = workspace.stage("dft-prepare")
    selection = json.loads((workspace.stage("selection") / "summary.json").read_text(encoding="utf-8"))
    count = int(selection["selected_count"])
    selected_dir = workspace.stage("selection") / "selected_poscars"
    sources = sorted(selected_dir.glob("POSCAR_*"))
    if len(sources) != count:
        raise ValueError(f"expected {count} selected POSCAR files, found {len(sources)}")
    common = stage / "common"
    common.mkdir(exist_ok=True)
    shutil.copy2(workspace.config.dft["incar"], common / "INCAR")
    kpoints = workspace.config.dft.get("kpoints")
    if isinstance(kpoints, Path) and kpoints.is_file():
        shutil.copy2(kpoints, common / "KPOINTS")
    titles = assemble_potcar(workspace, common / "POTCAR", validation["elements"])
    jobs = stage / "jobs"
    jobs.mkdir(exist_ok=True)
    atom_counts = set()
    for index, source in enumerate(sources, start=1):
        task = jobs / f"frame-{index:04d}"
        task.mkdir(exist_ok=True)
        if not (task / "POSCAR").exists():
            shutil.copy2(source, task / "POSCAR")
            shutil.copy2(common / "INCAR", task / "INCAR")
            if (common / "KPOINTS").exists():
                shutil.copy2(common / "KPOINTS", task / "KPOINTS")
            shutil.copy2(common / "POTCAR", task / "POTCAR")
        atoms = read(task / "POSCAR", format="vasp")
        atom_counts.add(len(atoms))
        (task / "task.json").write_text(json.dumps({
            "frame": index,
            "atoms": len(atoms),
            "calculation": "static VASP energy/forces/virial for DPA4 fine-tuning",
        }, indent=2) + "\n", encoding="utf-8")
    if len(atom_counts) != 1:
        raise ValueError(f"selected frames have inconsistent atom counts: {atom_counts}")
    result = {
        "status": "completed",
        "selected_count": count,
        "recommended_count": selection["recommended_count"],
        "atoms_per_frame": next(iter(atom_counts)),
        "total_tasks": count,
        "elements": validation["elements"],
        "potcar_titles": titles,
        "jobs_dir": str(jobs),
        "uses_kspacing": validation["uses_kspacing"],
        "bohrium_cores": int(workspace.config.dft["dispatcher"]["cores"]),
        "vasp_command": workspace.config.dft["dispatcher"]["command"],
    }
    (stage / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def submit_dft(workspace: RunWorkspace, *, wait: bool = False) -> dict[str, Any]:
    validate_dft_inputs(workspace, require_dispatcher=True)
    try:
        from dpdispatcher import Machine, Resources, Submission, Task
    except Exception as exc:
        raise ExternalProgramError(f"cannot import DPDispatcher: {type(exc).__name__}") from exc
    prepare_stage = workspace.stage("dft-prepare")
    jobs = sorted(path for path in (prepare_stage / "jobs").glob("frame-*") if path.is_dir())
    unfinished = [path for path in jobs if not (path / "OUTCAR").is_file()]
    if not unfinished:
        return {"status": "completed", "ready": len(jobs), "total": len(jobs), "n_submitted": 0}
    dispatcher = workspace.config.dft["dispatcher"]
    job_name = dynamic_job_name(workspace.config.name, workspace.root.name)
    try:
        machine_data = load_machine_safely(dispatcher["machine"])
        profile = machine_data.setdefault("remote_profile", {})
        # DPDispatcher authenticates from environment variables, but its
        # OpenAPI submission payload still reads project_id from the profile.
        if not profile.get("project_id") and os.environ.get("BOHRIUM_PROJECT_ID"):
            profile["project_id"] = int(os.environ["BOHRIUM_PROJECT_ID"])
        input_data = profile.get("input_data")
        if isinstance(input_data, dict):
            input_data["job_name"] = job_name
        else:
            profile["job_name"] = job_name
        machine = Machine.load_from_dict(machine_data)
        resources = Resources.load_from_json(str(dispatcher["resources"]))
        tasks = []
        for task_dir in unfinished:
            forward = [name for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR") if (task_dir / name).is_file()]
            tasks.append(Task(
                command=dispatcher["command"],
                task_work_path=str(task_dir.relative_to(prepare_stage)),
                forward_files=forward,
                backward_files=BACKWARD_FILES,
                outlog="vasp.stdout",
                errlog="vasp.stderr",
            ))
        submission = Submission(
            work_base=str(prepare_stage),
            machine=machine,
            resources=resources,
            task_list=tasks,
            forward_common_files=[],
            backward_common_files=[],
        )
        submission.generate_jobs()
        submission.submission_hash = submission.get_hash()
        returned = submission.run_submission(
            exit_on_submit=not wait,
            clean=dispatcher["clean_remote_after_success"],
        )
        manifest = prepare_stage / "submitted_tasks.json"
        if manifest.exists():
            manifest.chmod(stat.S_IRUSR | stat.S_IWUSR)
        ready = sum((path / "OUTCAR").is_file() for path in jobs)
        result = {
            "status": "completed" if ready == len(jobs) else "submitted",
            "submission_hash": submission.submission_hash,
            "ready": ready,
            "total": len(jobs),
            "n_submitted": len(tasks),
            "job_states": [str(getattr(job, "job_state", "unknown")) for job in submission.belonging_jobs],
            "waited": wait,
            "serialized_returned": isinstance(returned, dict),
            "job_name": job_name,
            "machine_cores": machine_core_count(machine_data),
            "command_cores": command_core_count(dispatcher["command"]),
        }
        return result
    except SystemExit:
        # Some DPDispatcher versions implement exit_on_submit with SystemExit.
        return {"status": "submitted", "ready": 0, "total": len(jobs), "n_submitted": len(tasks)}
    except Exception as exc:
        # Do not persist provider exceptions: they may echo credential-bearing config.
        raise ExternalProgramError(
            f"DPDispatcher submit/resume failed ({type(exc).__name__}); inspect local dpdispatcher.log"
        ) from exc


def local_dft_status(workspace: RunWorkspace) -> dict[str, Any]:
    jobs_dir = workspace.stage("dft-prepare") / "jobs"
    jobs = sorted(path for path in jobs_dir.glob("frame-*") if path.is_dir()) if jobs_dir.exists() else []
    ready = sum((path / "OUTCAR").is_file() for path in jobs)
    return {"ready": ready, "total": len(jobs), "jobs_dir": str(jobs_dir)}
