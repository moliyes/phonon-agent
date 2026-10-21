from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from phonon_agent.config import load_config
from phonon_agent.dft import prepare_dft, submit_dft
from phonon_agent.workspace import create_or_resume


def test_prepare_dft_preserves_incar_and_potcar_order(case: Path):
    config = load_config(case)
    workspace = create_or_resume(config)
    selection = workspace.stage("selection")
    poscars = selection / "selected_poscars"
    poscars.mkdir()
    shutil.copy2(case / "inputs" / "POSCAR", poscars / "POSCAR_0001")
    (selection / "summary.json").write_text(json.dumps({
        "selected_count": 1,
        "recommended_count": 1,
    }))
    result = prepare_dft(workspace)
    task = workspace.stage("dft-prepare") / "jobs" / "frame-0001"
    assert (task / "INCAR").read_bytes() == (case / "inputs" / "INCAR").read_bytes()
    potcar = (task / "POTCAR").read_text()
    assert potcar.index("Ag_pv") < potcar.index("PAW_PBE I")
    assert result["elements"] == ["Ag", "I"]


@pytest.mark.parametrize(("wait", "exit_on_submit"), [(False, True), (True, False)])
def test_submit_uses_dpdispatcher_wait_mode(
    case: Path, monkeypatch, wait: bool, exit_on_submit: bool
):
    (case / "machine.json").write_text(json.dumps({
        "batch_type": "OpenAPI",
        "context_type": "OpenAPIContext",
        "remote_profile": {"machine_type": "c16_m64_cpu"},
    }))
    (case / "machine.json").chmod(0o600)
    monkeypatch.setenv("BOHRIUM_ACCESS_KEY", "environment-only-secret")
    monkeypatch.setenv("BOHRIUM_PROJECT_ID", "99")
    config = load_config(case)
    workspace = create_or_resume(config)
    selection = workspace.stage("selection")
    poscars = selection / "selected_poscars"
    poscars.mkdir()
    shutil.copy2(case / "inputs" / "POSCAR", poscars / "POSCAR_0001")
    (selection / "summary.json").write_text(json.dumps({"selected_count": 1, "recommended_count": 1}))
    prepare_dft(workspace)
    calls = {}

    class Loader:
        @classmethod
        def load_from_json(cls, path):
            return cls()

        @classmethod
        def load_from_dict(cls, value):
            calls["machine"] = value
            return cls()

    class Task:
        def __init__(self, **kwargs):
            calls.setdefault("tasks", []).append(kwargs)

    class Submission:
        def __init__(self, **kwargs):
            self.belonging_jobs = [types.SimpleNamespace(job_state="submitted")]

        def generate_jobs(self):
            pass

        def get_hash(self):
            return "hash-1"

        def run_submission(self, **kwargs):
            calls["run"] = kwargs
            return {}

    fake = types.SimpleNamespace(Machine=Loader, Resources=Loader, Task=Task, Submission=Submission)
    monkeypatch.setitem(sys.modules, "dpdispatcher", fake)
    result = submit_dft(workspace, wait=wait)
    assert result["status"] == "submitted"
    assert calls["run"]["exit_on_submit"] is exit_on_submit
    assert calls["tasks"][0]["forward_files"] == ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]
    assert calls["machine"]["remote_profile"]["job_name"] == result["job_name"]
    assert calls["machine"]["remote_profile"]["project_id"] == 99
    assert "access_key" not in calls["machine"]["remote_profile"]
    assert result["machine_cores"] == result["command_cores"] == 16
