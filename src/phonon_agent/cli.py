from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ase.io import read

from . import __version__
from .config import load_config
from .dft import local_dft_status
from .errors import PhononAgentError
from .private_config import check_private_config, run_setup
from .stage_runner import EXECUTABLE_STAGES, resume_workspace, run_stage
from .state import StateStore
from .templates import init_case
from .validation import assert_valid, validate_config
from .workspace import RunWorkspace, latest_run


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _resolve_run(target: Path) -> tuple[Path, object]:
    target = target.expanduser().resolve()
    if (target / "state.json").is_file():
        state = StateStore(target / "state.json").load()
        return target, load_config(Path(state["source_config"]))
    config = load_config(target)
    run = latest_run(config)
    if run is None:
        raise PhononAgentError(f"no runs found for {config.name}")
    return run, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonon-agent")
    parser.add_argument("--version", action="version", version=f"phonon-agent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="configure private Bohrium and POTCAR defaults")
    setup.add_argument("--check", action="store_true", help="check configuration without showing secrets")
    init = sub.add_parser("init", help="create a standard case workspace")
    init.add_argument("case", type=Path)
    init.add_argument("--profile", choices=("agi",))
    validate = sub.add_parser("validate", help="validate configuration and local dependencies")
    validate.add_argument("case", type=Path)
    validate.add_argument("--until", choices=EXECUTABLE_STAGES, default="dft-collect")
    validate.add_argument("--no-dispatcher", action="store_true")
    plan = sub.add_parser("plan", help="show local and remote work estimates without executing")
    plan.add_argument("case", type=Path)
    stage = sub.add_parser("stage", help="run exactly one editable workspace stage")
    stage.add_argument("case", type=Path)
    stage.add_argument("stage", choices=EXECUTABLE_STAGES)
    stage.add_argument("--new", action="store_true", help="start a new run (phonon-check only)")
    stage.add_argument("--wait", action="store_true", help="wait for VASP completion when submitting")
    status = sub.add_parser("status", help="show local workflow status")
    status.add_argument("target", type=Path)
    resume = sub.add_parser("resume", help="resume a concrete run and refresh DPDispatcher")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--wait", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "setup":
            result = check_private_config() if args.check else run_setup()
            _json(result)
            return 0 if result.get("ok", result.get("configured", False)) else 2
        if args.command == "init":
            print(init_case(args.case, profile=args.profile))
            return 0
        if args.command == "validate":
            config = load_config(args.case)
            until = (
                "dft-prepare"
                if args.no_dispatcher
                and EXECUTABLE_STAGES.index(args.until)
                >= EXECUTABLE_STAGES.index("dft-submit")
                else args.until
            )
            report = validate_config(config, until=until)
            _json(report)
            return 0 if report["ok"] else 2
        if args.command == "plan":
            config = load_config(args.case)
            atoms = read(config.structure, format="vasp")
            repeat = config.md["repeat"]
            frames_per_temperature = config.md["production_steps"] // config.md["save_interval_steps"] + 1
            _json({
                "project": config.name,
                "unitcell_atoms": len(atoms),
                "md_atoms": len(atoms) * repeat[0] * repeat[1] * repeat[2],
                "md_ensemble": config.md["ensemble"],
                "md_temperatures_k": config.md["temperatures_k"],
                "candidate_frames_estimate": frames_per_temperature * len(config.md["temperatures_k"]) + int(config.selection["include_initial_supercell"]),
                "entropy_scan_max": config.selection["max_structures"],
                "selection_count": config.selection["count"] or "automatic: first >=95% of scanned maximum entropy",
                "remote_vasp_limit": None,
                "remote_submit_requires_until": "dft-submit",
                "dft_collection_format": "deepmd/npy",
            })
            return 0
        if args.command == "stage":
            config = load_config(args.case)
            workspace = run_stage(
                config,
                args.stage,
                force_new=args.new,
                wait=args.wait,
            )
            _json({"run": str(workspace.root), "state": workspace.store.load()})
            return 2 if workspace.store.load()["status"] == "waiting_user" else 0
        if args.command == "resume":
            workspace = resume_workspace(args.run_dir, wait=args.wait)
            _json({"run": str(workspace.root), "state": workspace.store.load()})
            return 2 if workspace.store.load()["status"] == "waiting_user" else 0
        if args.command == "status":
            run_dir, config = _resolve_run(args.target)
            workspace = RunWorkspace(run_dir, config)
            _json({
                "run": str(run_dir),
                "state": workspace.store.load(),
                "dft_local": local_dft_status(workspace),
            })
            return 0
    except (PhononAgentError, FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
