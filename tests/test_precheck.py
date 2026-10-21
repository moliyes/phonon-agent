from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from phonon_agent.config import load_config
from phonon_agent.errors import WaitingForUser
from phonon_agent.precheck import run_precheck
from phonon_agent.workspace import create_or_resume


def test_precheck_pauses_below_threshold(case: Path):
    config_path = case / "workflow.yaml"
    raw = yaml.safe_load(config_path.read_text())
    raw["precheck"]["enabled"] = True
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    workspace = create_or_resume(load_config(case))
    summary = workspace.stage("phonon-check") / "ph-runs" / "precheck-001" / "results" / "pretrained" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"min_frequency_thz": -0.2}))
    with pytest.raises(WaitingForUser, match="below"):
        run_precheck(workspace)
