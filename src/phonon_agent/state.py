from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_json, load_json, utc_now


STAGES = (
    "phonon-check",
    "md",
    "entropy",
    "selection",
    "dft-prepare",
    "dft-submit",
    "dft-collect",
    "training",
    "properties",
)


@dataclass
class StateStore:
    path: Path

    def load(self) -> dict[str, Any]:
        return load_json(self.path)

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.path, state)

    def update_stage(self, name: str, status: str, **details: Any) -> dict[str, Any]:
        state = self.load()
        current = dict(state["stages"].get(name) or {})
        current.update(details)
        current["status"] = status
        current["updated_at"] = utc_now()
        state["stages"][name] = current
        terminal = [state["stages"][key]["status"] for key in STAGES[:6]]
        if any(item == "failed" for item in terminal):
            state["status"] = "failed"
        elif any(item == "waiting_user" for item in terminal):
            state["status"] = "waiting_user"
        elif status == "submitted":
            state["status"] = "waiting_dft"
        else:
            state["status"] = "running"
        self.save(state)
        return state


def initial_state(name: str, fingerprint: str, config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "project": name,
        "status": "created",
        "config_fingerprint": fingerprint,
        "source_config": str(config_path),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "stages": {name: {"status": "pending"} for name in STAGES},
        "errors": [],
    }
