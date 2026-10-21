from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory

from .md import temperature_tag
from .workspace import RunWorkspace


def load_pool(workspace: RunWorkspace):
    config = workspace.config
    md_stage = workspace.stage("md")
    frames = []
    rows: list[dict[str, Any]] = []
    if config.selection["include_initial_supercell"]:
        atoms = read(md_stage / "supercell.POSCAR", format="vasp")
        atoms.calc = None
        frames.append(atoms)
        rows.append({"pool_index": 0, "temperature_K": 0.0, "trajectory": "initial_supercell", "trajectory_frame": -1})
    stride = int(config.selection["pool_stride"])
    for temperature in config.md["temperatures_k"]:
        path = md_stage / f"md_{temperature_tag(temperature)}.traj"
        with Trajectory(str(path), "r") as trajectory:
            for frame_index, source in enumerate(trajectory):
                if frame_index % stride:
                    continue
                atoms = source.copy()
                atoms.calc = None
                if atoms.has("momenta"):
                    atoms.set_momenta(None)
                frames.append(atoms)
                rows.append({
                    "pool_index": len(frames) - 1,
                    "temperature_K": float(temperature),
                    "trajectory": path.name,
                    "trajectory_frame": frame_index,
                })
    if not frames:
        raise ValueError("candidate pool is empty")
    counts = {len(frame) for frame in frames}
    compositions = {tuple(sorted(frame.symbols.formula.count().items())) for frame in frames}
    if len(counts) != 1 or len(compositions) != 1:
        raise ValueError("candidate pool has inconsistent atom counts or compositions")
    return frames, rows


class EntropyBackend:
    def __init__(self, descriptors: np.ndarray, device: str, h: float, batch_size: int):
        self.h = h
        self.batch_size = batch_size
        self.gpu = False
        if device.startswith("cuda"):
            try:
                import torch
                from quests.gpu.entropy import delta_entropy, entropy
                if torch.cuda.is_available():
                    self.values = torch.as_tensor(descriptors, dtype=torch.float32, device=device)
                    self.torch = torch
                    self.entropy_fn = entropy
                    self.delta_fn = delta_entropy
                    self.device = device
                    self.gpu = True
                    return
            except Exception:
                pass
        from quests.entropy import delta_entropy, entropy
        self.values = descriptors
        self.torch = None
        self.entropy_fn = entropy
        self.delta_fn = delta_entropy
        self.device = "cpu"

    def subset(self, indices: list[int]):
        if self.gpu:
            idx = self.torch.as_tensor(indices, dtype=self.torch.long, device=self.device)
            return self.values.index_select(0, idx).reshape(-1, self.values.shape[-1])
        return self.values[np.asarray(indices, dtype=int)].reshape(-1, self.values.shape[-1])

    def entropy(self, indices: list[int]) -> float:
        values = self.subset(indices)
        kwargs = {"h": self.h, "batch_size": self.batch_size}
        if self.gpu:
            return float(self.entropy_fn(values, device=self.device, **kwargs).detach().cpu())
        return float(self.entropy_fn(values, **kwargs))

    def gains(self, remaining: list[int], selected: list[int], natoms: int) -> np.ndarray:
        kwargs = {"h": self.h, "batch_size": self.batch_size}
        result = self.delta_fn(
            self.subset(remaining), self.subset(selected),
            **({"device": self.device, **kwargs} if self.gpu else kwargs),
        )
        if self.gpu:
            result = result.detach().cpu().numpy()
        return np.asarray(result, dtype=float).reshape(len(remaining), natoms).sum(axis=1)


def recommended_count(
    entropies: list[float], fraction: float, counts: list[int] | None = None
) -> int:
    maximum = max(entropies)
    threshold = fraction * maximum
    index = next(index for index, value in enumerate(entropies) if value >= threshold)
    return counts[index] if counts is not None else index + 1


def run_entropy(workspace: RunWorkspace) -> dict:
    stage = workspace.stage("entropy")
    frames, rows = load_pool(workspace)
    selection = workspace.config.selection
    os.environ.setdefault("NUMBA_CACHE_DIR", str(stage / ".numba-cache"))
    from quests.descriptor import get_descriptors
    descriptors = np.asarray(
        get_descriptors(frames, k=selection["k"], cutoff=selection["cutoff_angstrom"], dtype="float32"),
        dtype=np.float32,
    )
    natoms = len(frames[0])
    if descriptors.shape[0] != len(frames) * natoms:
        raise ValueError(f"QUESTS descriptor shape mismatch: {descriptors.shape}")
    descriptors = descriptors.reshape(len(frames), natoms, -1)
    backend = EntropyBackend(
        descriptors,
        workspace.config.device,
        selection["bandwidth_inv_angstrom"],
        selection["batch_size"],
    )
    limit = min(int(selection["max_structures"]), len(frames))
    selected = [0] if selection["include_initial_supercell"] else [int(np.random.default_rng(42).integers(len(frames)))]
    remaining = [index for index in range(len(frames)) if index not in selected]
    entropies = [backend.entropy(selected)]
    while len(selected) < limit:
        scores = backend.gains(remaining, selected, natoms)
        position = int(np.argmax(scores))
        selected.append(remaining.pop(position))
        entropies.append(backend.entropy(selected))
    fraction = float(selection["entropy_fraction"])
    recommendation = recommended_count(entropies, fraction)
    maximum_index = int(np.argmax(entropies))
    with (stage / "entropy_saturation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["selected_structure_count", "entropy", "fraction_of_maximum", "recommended"])
        writer.writeheader()
        maximum = max(entropies)
        for index, value in enumerate(entropies, start=1):
                writer.writerow({
                    "selected_structure_count": index,
                    "entropy": value,
                    "fraction_of_maximum": value / maximum if maximum else 1.0,
                    "recommended": index == recommendation,
                })
    with (stage / "entropy_selection_order.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["selection_rank", *rows[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, pool_index in enumerate(selected, start=1):
            writer.writerow({"selection_rank": rank, **rows[pool_index]})
    summary = {
        "status": "completed",
        "candidate_pool_size": len(frames),
        "scanned_count": limit,
        "maximum_entropy": max(entropies),
        "maximum_at_count": maximum_index + 1,
        "maximum_at_scan_boundary": maximum_index + 1 == limit,
        "entropy_fraction": fraction,
        "recommended_count": recommendation,
        "device": backend.device,
    }
    (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
