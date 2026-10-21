from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from phonon_agent.templates import init_case


POSCAR = """AgI
1.0
4 0 0
0 4 0
0 0 6
Ag I
1 1
Direct
0 0 0
0.5 0.5 0.5
"""


@pytest.fixture
def case(tmp_path: Path) -> Path:
    root = tmp_path / "case"
    init_case(root)
    inputs = root / "inputs"
    (inputs / "POSCAR").write_text(POSCAR)
    (inputs / "INCAR").write_text("IBRION=-1\nNSW=0\nKSPACING=0.25\n")
    (inputs / "KPOINTS").write_text("mesh\n0\nGamma\n2 2 2\n0 0 0\n")
    archive = root / "potpaw54.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("pbe/Ag_pv/POTCAR", " TITEL = PAW_PBE Ag_pv 01Jan2000\n")
        handle.writestr("pbe/I/POTCAR", " TITEL = PAW_PBE I 01Jan2000\n")
    (root / "machine.json").write_text(json.dumps({
        "batch_type": "Bohrium",
        "context_type": "BohriumContext",
        "remote_profile": {
            "email": "user@example.invalid",
            "password": "do-not-copy",
            "project_id": 1,
            "input_data": {"scass_type": "c16_m64_cpu"},
        },
    }))
    (root / "machine.json").chmod(0o600)
    (root / "resources.json").write_text(json.dumps({"group_size": 1}))
    workflow = {
        "schema_version": 2,
        "project": {"name": "test", "runs_dir": "runs"},
        "scripts": {
            "phonon-check": "scripts/00_phonon_check.py",
            "md": "scripts/10_run_md.py",
            "entropy": "scripts/20_entropy_prescreen.py",
            "selection": "scripts/30_select_frames.py",
            "dft-prepare": "scripts/40_prepare_dft.py",
            "dft-submit": "scripts/50_submit_dft.py",
            "dft-collect": "scripts/60_collect_dft.py",
        },
        "structure": {"file": "inputs/POSCAR"},
        "tools": {"ph_command": "/bin/true"},
        "model": {"pretrained": "builtin:dpa4", "device": "cpu"},
        "environments": {"deepmd": {"python": sys.executable}},
        "precheck": {"enabled": False, "supercell": [1, 1, 1], "mesh": [2, 2, 2], "imaginary_stop_thz": -0.1},
        "md": {"ensemble": "nvt", "repeat": [1, 1, 1], "temperatures_k": [50], "equilibration_steps": 1, "production_steps": 2, "save_interval_steps": 1},
        "selection": {"count": None, "max_structures": 3, "entropy_fraction": 0.95},
        "dft": {
            "incar": "inputs/INCAR",
            "kpoints": "inputs/KPOINTS",
            "potcar": {"archive": "potpaw54.zip", "family": "pbe", "variants": {"Ag": "Ag_pv", "I": "I"}},
            "dispatcher": {
                "machine": "machine.json",
                "resources": "resources.json",
                "cores": 16,
                "command": "mpirun -n 16 vasp_std",
            },
        },
    }
    (root / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False))
    return root
