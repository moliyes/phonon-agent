"""Compatibility imports for the script-driven stage runner.

Batch execution was intentionally removed. Call :func:`run_stage` once for
each stage so its result can be inspected before continuing.
"""

from __future__ import annotations

from .errors import ConfigError
from .stage_runner import EXECUTABLE_STAGES, resume_workspace, run_stage

__all__ = ["EXECUTABLE_STAGES", "resume_workspace", "run_stage"]


def run_workflow(*args, **kwargs):
    raise ConfigError(
        "batch run_workflow was removed; use phonon_agent.stage_runner.run_stage "
        "for one stage at a time"
    )
