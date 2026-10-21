from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from phonon_agent.config import load_config
from phonon_agent.dft import prepare_dft
from phonon_agent.private_config import (
    PRIVATE_CONFIG_GUIDANCE,
    check_private_config,
    configure_compute_environment,
    dynamic_job_name,
    run_setup,
)
from phonon_agent.validation import validate_config
from phonon_agent.workspace import create_or_resume


def _answers(monkeypatch, values: list[str], secrets: list[str]) -> None:
    iterator = iter(values)
    secret_iterator = iter(secrets)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(iterator))
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(secret_iterator))


def _mock_compute(monkeypatch) -> None:
    python = Path(sys.executable).resolve()
    monkeypatch.setattr(
        "phonon_agent.private_config.configure_compute_environment",
        lambda value: {
            "python": python,
            "deepmd_version": "test",
            "phonon_agent_command": python.parent / "phonon-agent",
            "ph_command": python.parent / "ph",
        },
    )


def test_setup_legacy_is_private_and_repeat_creates_backups(
    case: Path, tmp_path: Path, monkeypatch
):
    private = tmp_path / "private"
    monkeypatch.setenv("PHONON_AGENT_CONFIG_DIR", str(private))
    _mock_compute(monkeypatch)
    values = [str(Path(sys.executable).parent.parent), "1", "user@example.invalid", "42", str(case / "potpaw54.zip")]
    _answers(monkeypatch, values, ["new-secret", "new-secret"])
    result = run_setup()
    assert result["auth_mode"] == "legacy-password"
    assert result["potcar_family"] == "pbe"
    assert stat_mode(private) == 0o700
    assert stat_mode(private / "config.yaml") == 0o600
    assert stat_mode(private / "bohrium" / "machine.json") == 0o600
    machine_text = (private / "bohrium" / "machine.json").read_text()
    assert "new-secret" in machine_text
    assert "new-secret" not in json.dumps(result)

    _answers(monkeypatch, values, ["replacement", "replacement"])
    second = run_setup()
    assert len(second["backups_created"]) == 3
    assert all(Path(path).is_file() for path in second["backups_created"])


def test_setup_openapi_environment_and_safe_check(case: Path, tmp_path: Path, monkeypatch):
    private = tmp_path / "private"
    monkeypatch.setenv("PHONON_AGENT_CONFIG_DIR", str(private))
    monkeypatch.setenv("BOHRIUM_ACCESS_KEY", "private-access-key")
    monkeypatch.setenv("BOHRIUM_PROJECT_ID", "42")
    _mock_compute(monkeypatch)
    _answers(monkeypatch, [str(Path(sys.executable).parent.parent), "3", str(case / "potpaw54.zip")], [])
    run_setup()
    report = check_private_config()
    serialized = json.dumps(report)
    assert report["ok"]
    assert report["authentication"] == "openapi-environment"
    assert report["machine_cores"] == report["command_cores"] == 16
    assert "private-access-key" not in serialized


def test_global_defaults_and_project_override(case: Path, tmp_path: Path, monkeypatch):
    private = tmp_path / "private"
    bohrium = private / "bohrium"
    bohrium.mkdir(parents=True)
    machine = bohrium / "machine.json"
    resources = bohrium / "resources.json"
    machine.write_text(json.dumps({
        "remote_profile": {
            "email": "x@example.invalid",
            "password": "secret",
            "project_id": 1,
            "input_data": {"scass_type": "c16_m64_cpu"},
        }
    }))
    resources.write_text('{"group_size": 1}\n')
    machine.chmod(0o600)
    resources.chmod(0o600)
    (private / "config.yaml").write_text(yaml.safe_dump({
        "schema_version": 1,
        "environments": {"deepmd": {"python": sys.executable}},
        "tools": {"phonon_kit": {"command": str(Path(sys.executable).parent / "ph")}},
        "potcar": {"source": str(case / "potpaw54.zip"), "family": "pbe"},
        "bohrium": {
            "machine": "bohrium/machine.json",
            "resources": "bohrium/resources.json",
            "cores": 16,
            "command_template": "mpirun -n {cores} vasp_std",
        },
    }))
    monkeypatch.setenv("PHONON_AGENT_CONFIG_DIR", str(private))
    workflow_path = case / "workflow.yaml"
    raw = yaml.safe_load(workflow_path.read_text())
    raw.pop("environments")
    raw.pop("tools")
    raw["dft"]["potcar"] = {"variants": {"Ag": "Ag_pv", "I": "I"}}
    raw["dft"]["dispatcher"] = {"clean_remote_after_success": False}
    workflow_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(case)
    assert config.dft["potcar"]["source"] == (case / "potpaw54.zip").resolve()
    assert config.dft["dispatcher"]["machine"] == machine.resolve()
    assert config.dft["dispatcher"]["command"] == "mpirun -n 16 vasp_std"
    assert config.deepmd_python == Path(sys.executable).resolve()
    assert config.ph_command == str(Path(sys.executable).parent / "ph")

    override = case / "other.zip"
    shutil.copy2(case / "potpaw54.zip", override)
    raw["dft"]["potcar"]["source"] = "other.zip"
    workflow_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert load_config(case).dft["potcar"]["source"] == override.resolve()


def test_directory_potcar_source(case: Path):
    source = case / "potcars"
    (source / "Ag_pv").mkdir(parents=True)
    (source / "I").mkdir()
    (source / "Ag_pv" / "POTCAR").write_text(" TITEL = PAW_PBE Ag_pv 01Jan2000\n")
    (source / "I" / "POTCAR").write_text(" TITEL = PAW_PBE I 01Jan2000\n")
    path = case / "workflow.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["dft"]["potcar"].pop("archive")
    raw["dft"]["potcar"]["source"] = "potcars"
    raw["dft"]["potcar"]["family"] = ""
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
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
    assert result["potcar_titles"] == ["PAW_PBE Ag_pv 01Jan2000", "PAW_PBE I 01Jan2000"]


def test_stage_aware_validation_and_core_mismatch(case: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PHONON_AGENT_CONFIG_DIR", str(tmp_path / "absent"))
    path = case / "workflow.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["dft"]["potcar"].pop("archive")
    raw["dft"]["dispatcher"].pop("machine")
    raw["dft"]["dispatcher"].pop("resources")
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(case)
    early = validate_config(config, until="entropy")
    assert early["ok"]
    prepare = validate_config(config, until="dft-prepare")
    assert not prepare["ok"]
    assert any(PRIVATE_CONFIG_GUIDANCE in error for error in prepare["errors"])

    raw = yaml.safe_load(path.read_text())
    raw["dft"]["potcar"]["source"] = "potpaw54.zip"
    raw["dft"]["potcar"]["family"] = "pbe"
    raw["dft"]["dispatcher"].update({
        "machine": "machine.json",
        "resources": "resources.json",
        "cores": 16,
        "command": "mpirun -n 32 vasp_std",
    })
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    mismatch = validate_config(load_config(case), until="dft-submit")
    assert not mismatch["ok"]
    assert any("-n 16" in error for error in mismatch["errors"])


def test_dynamic_job_name_is_stable_and_bounded():
    short = dynamic_job_name("AgI", "AgI-001")
    assert short == "pa-AgI-AgI-001-vasp"
    long_name = dynamic_job_name("x" * 100, "run-001")
    assert len(long_name) <= 63
    assert long_name == dynamic_job_name("x" * 100, "run-001")


def test_compute_environment_uses_python_and_creates_global_commands(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv("PHONON_AGENT_BIN_DIR", str(bin_dir))
    result = configure_compute_environment(Path(sys.executable).parent.parent)
    assert result["python"].resolve() == Path(sys.executable).resolve()
    assert (bin_dir / "phonon-agent").resolve() == Path(sys.executable).parent / "phonon-agent"
    assert (bin_dir / "ph").resolve() == Path(sys.executable).parent / "ph"


def test_setup_without_checkouts_installs_from_published_sources(tmp_path: Path, monkeypatch):
    from phonon_agent import private_config

    environment = tmp_path / "environment" / "bin"
    environment.mkdir(parents=True)
    python = environment / "python"
    monkeypatch.setenv("PHONON_AGENT_BIN_DIR", str(tmp_path / "commands"))
    monkeypatch.setattr(private_config, "normalize_python_path", lambda value: python)
    monkeypatch.setattr(private_config, "_project_roots", lambda: (None, None))
    monkeypatch.setattr(private_config, "probe_python_module", lambda *args: (True, "test"))
    requirements = []

    def fake_install(interpreter, requirement):
        assert interpreter == python
        requirements.append(requirement)
        command = "ph" if requirement == "dpa-phonon" else "phonon-agent"
        (environment / command).touch()

    monkeypatch.setattr(private_config, "_install_package", fake_install)
    result = configure_compute_environment(python)

    assert requirements == [
        "phonon-agent[sampling,conversion] @ git+https://github.com/moliyes/phonon-agent.git",
        "dpa-phonon",
    ]
    assert result["phonon_agent_command"].resolve() == environment / "phonon-agent"
    assert result["ph_command"].resolve() == environment / "ph"


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777
