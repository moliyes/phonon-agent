from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from ase.calculators.calculator import Calculator, all_changes

from phonon_agent.config import load_config
from phonon_agent.stage_runner import run_stage


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions)
        self.results = {
            "energy": float(1.0e-4 * np.sum(positions**2)),
            "forces": -2.0e-4 * positions,
            "stress": np.zeros(6),
        }


def test_short_md_to_dft_prepare_pipeline(case: Path, monkeypatch):
    path = case / "workflow.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["selection"].update({"max_structures": 3, "k": 3, "cutoff_angstrom": 3.0})
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    import deepmd.calculator
    monkeypatch.setattr(deepmd.calculator, "DP", lambda **kwargs: HarmonicCalculator())
    config = load_config(case)
    workspace = None
    for stage in ("phonon-check", "md", "entropy"):
        workspace = run_stage(config, stage)
    raw = yaml.safe_load(path.read_text())
    raw["selection"]["count"] = 1
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(case)
    for stage in ("selection", "dft-prepare"):
        workspace = run_stage(config, stage)
    assert workspace is not None
    state = workspace.store.load()
    assert state["status"] == "ready"
    assert state["next_stage"] == "dft-submit"
    assert state["stages"]["dft-prepare"]["status"] == "completed"
    assert state["stages"]["entropy"]["recommended_count"] >= 1
    assert (workspace.stage("dft-prepare") / "jobs" / "frame-0001" / "POTCAR").is_file()
