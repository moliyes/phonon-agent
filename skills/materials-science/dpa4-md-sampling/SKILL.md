---
name: dpa4-md-sampling
description: Run DPA4 MD and QUESTS entropy frame selection.
version: 0.3.0
author: moliyes
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [materials-science, DPA4, molecular-dynamics, QUESTS]
    category: materials-science
    related_skills: [phonon-agent]
    requires_tools: [terminal]
---

# DPA4 MD Sampling

This workflow explores finite-temperature structures with DPA4, uses QUESTS to
choose representative candidates, and prepares them for true DFT labels. DPA4
energies and forces are never training labels.

For a new task, run `phonon-agent init <case>`. Use `--profile agi` only
when the user explicitly wants the validated AgI profile. Replace the
placeholder `inputs/POSCAR`, then inspect `workflow.yaml` and
`scripts/README.md`.

Before MD, verify the user-configured DeepMD Python with
`phonon-agent setup --check`. Never install or upgrade DeepMD-kit automatically.
Run through the global `phonon-agent` command so MD and phonon-kit use that same
environment; do not rely on the caller's activated shell environment.
For `builtin:dpa4`, prefer a machine-local model made with `ph model freeze`;
the packaged `.pt` is the fallback, not a portable pre-frozen `.pt2`.

1. Inspect structure size, requested ensemble, temperatures, repeat, time step,
   equilibration, production length, save interval, seed, and safety limits.
2. Keep the user's explicit scientific settings. For unspecified settings, use
   an existing validated profile when one matches; otherwise explain the chosen
   values in `workflow.yaml`.
3. Execute one stage at a time with `phonon-agent stage <case> <stage>`.
   After MD, inspect its state, log, trajectories, and safety diagnostics before
   running entropy. Never use the removed batch `run --until` interface.
4. After entropy, read the current attempt's `summary.json` and report scan
   size, maximum location, boundary warning, and 95% recommendation.
5. Run selection only after reviewing entropy. If the user set
   `selection.count`, report the override and the recommendation.
6. Treat a safety stop or missing trajectory as a real failure. Do not fabricate
   a smaller successful trajectory.
7. Prefer `workflow.yaml` changes. Edit the failed stage script only for a
   concrete technical issue that configuration cannot solve. Technical
   compatibility fixes may proceed with an explanation; changing physics,
   selection logic, or weakening safety limits requires user approval. Never
   edit state, resolved config, successful outputs, or script snapshots.
