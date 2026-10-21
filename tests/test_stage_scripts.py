from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phonon_agent.config import load_config
from phonon_agent.errors import ConfigError
from phonon_agent.stage_runner import run_stage
from phonon_agent.templates import SCRIPT_SOURCES, init_case
from phonon_agent.validation import validate_config


def test_init_creates_editable_stage_scripts(tmp_path: Path):
    case = init_case(tmp_path / "new-case")
    raw = yaml.safe_load((case / "workflow.yaml").read_text())
    assert raw["schema_version"] == 2
    assert set(raw["scripts"]) == {
        "phonon-check", "md", "entropy", "selection",
        "dft-prepare", "dft-submit", "dft-collect",
    }
    for filename in SCRIPT_SOURCES:
        script = case / "scripts" / filename
        assert script.is_file()
        assert script.stat().st_mode & 0o111
        text = script.read_text()
        assert "def run(context):" in text
        assert len(text.splitlines()) > 40


def test_stage_requires_prerequisite(case: Path):
    with pytest.raises(ConfigError, match="start with|prerequisite"):
        run_stage(load_config(case), "md")


def test_missing_collect_script_only_blocks_collect_stage(case: Path):
    (case / "scripts" / "60_collect_dft.py").unlink()
    config = load_config(case)
    assert validate_config(config, until="dft-submit")["ok"]
    report = validate_config(config, until="dft-collect")
    assert not report["ok"]
    assert any("60_collect_dft.py" in error for error in report["errors"])


def test_failed_stage_can_retry_modified_script_in_same_run(case: Path):
    script = case / "scripts" / "00_phonon_check.py"
    script.write_text(
        "def run(context):\n    raise RuntimeError('first failure')\n",
        encoding="utf-8",
    )
    config = load_config(case)
    with pytest.raises(RuntimeError, match="first failure"):
        run_stage(config, "phonon-check")
    script.write_text(
        "def run(context):\n    return {'status': 'skipped', 'reason': 'test'}\n",
        encoding="utf-8",
    )
    workspace = run_stage(load_config(case), "phonon-check")
    attempts = workspace.store.load()["stages"]["phonon-check"]["attempts"]
    assert [item["status"] for item in attempts] == ["failed", "skipped"]
    assert attempts[0]["script_sha256"] != attempts[1]["script_sha256"]
    assert all(Path(item["script"]).is_file() for item in attempts)


def test_future_config_change_allowed_but_completed_input_change_rejected(case: Path):
    script = case / "scripts" / "00_phonon_check.py"
    script.write_text(
        "def run(context):\n    return {'status': 'skipped', 'reason': 'test'}\n",
        encoding="utf-8",
    )
    run_stage(load_config(case), "phonon-check")
    (case / "scripts" / "10_run_md.py").write_text(
        "def run(context):\n    raise RuntimeError('md sentinel')\n",
        encoding="utf-8",
    )
    path = case / "workflow.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["selection"]["count"] = 2
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    # A future-only selection change does not invalidate phonon-check.
    with pytest.raises(RuntimeError, match="md sentinel") as exc:
        run_stage(load_config(case), "md")
    assert "configuration now changes completed stage phonon-check" not in str(exc.value)

    raw = yaml.safe_load(path.read_text())
    raw["precheck"]["mesh"] = [3, 3, 3]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ConfigError, match="changes completed stage phonon-check"):
        run_stage(load_config(case), "md")
