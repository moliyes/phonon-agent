from __future__ import annotations

from pathlib import Path

import yaml

from phonon_agent.config import load_config
from phonon_agent.state import StateStore, initial_state
from phonon_agent.validation import validate_config
from phonon_agent.workspace import create_or_resume


def test_config_and_workspace_never_copy_machine_secret(case: Path):
    config = load_config(case)
    workspace = create_or_resume(config)
    resolved = (workspace.root / "config.resolved.yaml").read_text()
    state = (workspace.root / "state.json").read_text()
    assert "do-not-copy" not in resolved
    assert "do-not-copy" not in state
    assert not (workspace.root / "inputs" / "machine.json").exists()
    assert create_or_resume(config).root == workspace.root


def test_validation_requires_explicit_potcar_mapping(case: Path):
    path = case / "workflow.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["dft"]["potcar"]["variants"].pop("I")
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    report = validate_config(load_config(case))
    assert not report["ok"]
    assert any("I" in error and "POTCAR" in error for error in report["errors"])


def test_state_redacts_secret_keys(tmp_path: Path):
    path = tmp_path / "state.json"
    store = StateStore(path)
    state = initial_state("x", "abc", tmp_path / "workflow.yaml")
    state["password"] = "secret-value"
    store.save(state)
    assert "secret-value" not in path.read_text()
