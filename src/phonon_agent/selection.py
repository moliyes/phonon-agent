from __future__ import annotations

import csv
import json

from ase.io import write

from .entropy import load_pool
from .workspace import RunWorkspace


def run_selection(workspace: RunWorkspace) -> dict:
    entropy_stage = workspace.stage("entropy")
    stage = workspace.stage("selection")
    summary = json.loads((entropy_stage / "summary.json").read_text(encoding="utf-8"))
    count = workspace.config.selection["count"] or int(summary["recommended_count"])
    with (entropy_stage / "entropy_selection_order.csv").open(encoding="utf-8", newline="") as handle:
        order = list(csv.DictReader(handle))
    if count <= 0 or count > len(order):
        raise ValueError(f"selection count {count} is outside scanned range 1..{len(order)}")
    frames, _ = load_pool(workspace)
    selected_indices = [int(row["pool_index"]) for row in order[:count]]
    selected = [frames[index] for index in selected_indices]
    write(stage / f"selected_{count}.extxyz", selected, format="extxyz")
    poscars = stage / "selected_poscars"
    poscars.mkdir(exist_ok=True)
    for index, atoms in enumerate(selected, start=1):
        path = poscars / f"POSCAR_{index:04d}"
        if not path.exists():
            write(path, atoms, format="vasp", direct=True, vasp5=True, sort=False)
    result = {
        "status": "completed",
        "selected_count": count,
        "recommended_count": int(summary["recommended_count"]),
        "user_override": workspace.config.selection["count"] is not None,
        "selected_pool_indices": selected_indices,
        "poscar_dir": str(poscars),
    }
    (stage / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
