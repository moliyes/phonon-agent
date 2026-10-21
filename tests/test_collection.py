from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from phonon_agent.collection import collect_dft
from phonon_agent.config import load_config
from phonon_agent.errors import ConfigError
from phonon_agent.stage_runner import (
    EXECUTABLE_STAGES,
    run_stage,
    stage_input_fingerprint,
)
from phonon_agent.workspace import create_or_resume


class FakeLabeledSystem:
    def __init__(self, path: str):
        name = Path(path).parent.name
        if name == "bad":
            raise ValueError("unreadable")
        self.frames = 0 if name == "empty" else 1

    def __len__(self):
        return self.frames


class FakeMultiSystems:
    def __init__(self, *systems):
        self.systems = {"Ag1I1": systems[0]}

    def to(self, format_name: str, output: str):
        assert format_name == "deepmd/npy"
        target = Path(output) / "Ag1I1"
        target.mkdir(parents=True)
        (target / "type_map.raw").write_text("Ag\nI\n", encoding="utf-8")


def _fake_dpdata(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "dpdata",
        types.SimpleNamespace(
            LabeledSystem=FakeLabeledSystem,
            MultiSystems=FakeMultiSystems,
        ),
    )


def test_collect_converts_partial_success(case: Path, monkeypatch):
    _fake_dpdata(monkeypatch)
    workspace = create_or_resume(load_config(case))
    jobs = workspace.stage("dft-prepare") / "jobs"
    for name in ("good", "empty", "bad"):
        task = jobs / name
        task.mkdir(parents=True)
        (task / "OUTCAR").write_text(name, encoding="utf-8")

    result = collect_dft(workspace)
    assert result["outcar_count"] == 3
    assert result["successful_outcars"] == 1
    assert result["empty_outcars"] == 1
    assert result["failed_outcars"] == 1
    assert result["systems"] == ["Ag1I1"]
    assert result["total_frames"] == 1
    assert (Path(result["dpdata_dir"]) / "Ag1I1" / "type_map.raw").is_file()
    summary = json.loads(
        (workspace.stage("dft-collect") / "summary.json").read_text()
    )
    assert summary["empty_tasks"] == ["empty"]
    assert summary["failed_tasks"] == ["bad"]


def test_collect_fails_when_every_outcar_is_rejected(case: Path, monkeypatch):
    _fake_dpdata(monkeypatch)
    workspace = create_or_resume(load_config(case))
    jobs = workspace.stage("dft-prepare") / "jobs"
    for name in ("empty", "bad"):
        task = jobs / name
        task.mkdir(parents=True)
        (task / "OUTCAR").write_text(name, encoding="utf-8")

    with pytest.raises(ValueError, match="could not convert any OUTCAR"):
        collect_dft(workspace)
    assert not (workspace.stage("dft-collect") / "dpdata").exists()


def _mark_before_collect(workspace, config, submit_status: str) -> None:
    state = workspace.store.load()
    for stage in EXECUTABLE_STAGES[:-1]:
        state["stages"][stage].update(
            {
                "status": "completed",
                "input_fingerprint": stage_input_fingerprint(config, stage),
            }
        )
    state["stages"]["dft-submit"]["status"] = submit_status
    state["status"] = "waiting_dft" if submit_status == "submitted" else "ready"
    workspace.store.save(state)


def test_collect_waits_only_for_an_existing_submission(case: Path, monkeypatch):
    config = load_config(case)
    workspace = create_or_resume(config)
    _mark_before_collect(workspace, config, "submitted")
    (case / "scripts" / "60_collect_dft.py").write_text(
        "def run(context):\n"
        "    return {'status': 'completed', 'successful_outcars': 1, 'total_frames': 1}\n",
        encoding="utf-8",
    )
    calls = []

    def fake_wait(current_config, current_workspace):
        calls.append((current_config.path, current_workspace.root))
        state = current_workspace.store.load()
        state["stages"]["dft-submit"]["status"] = "completed"
        current_workspace.store.save(state)
        return current_workspace

    monkeypatch.setattr("phonon_agent.stage_runner._wait_for_submitted", fake_wait)
    result = run_stage(config, "dft-collect")
    assert calls == [(config.path, workspace.root)]
    assert result.store.load()["stages"]["dft-collect"]["status"] == "completed"


def test_collect_never_submits_a_pending_dft_stage(case: Path, monkeypatch):
    config = load_config(case)
    workspace = create_or_resume(config)
    _mark_before_collect(workspace, config, "pending")
    monkeypatch.setattr(
        "phonon_agent.stage_runner._wait_for_submitted",
        lambda *args: pytest.fail("pending jobs must not be submitted by dft-collect"),
    )
    with pytest.raises(ConfigError, match="prerequisite dft-submit is pending"):
        run_stage(config, "dft-collect")
