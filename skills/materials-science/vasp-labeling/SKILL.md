---
name: vasp-labeling
description: Prepare and submit static VASP training labels.
version: 0.5.0
author: moliyes
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [materials-science, VASP, DPDispatcher, Bohrium]
    category: materials-science
    related_skills: [phonon-agent, dpa4-md-sampling, dpa4-finetuning]
    requires_tools: [terminal]
---

# VASP Labeling

- Require `IBRION=-1` and `NSW=0`; preserve the complete user INCAR unchanged.
- If INCAR lacks `KSPACING`, require a user-confirmed KPOINTS file.
- Require an explicit POTCAR variant for every element on every task. Assemble
  in first-occurrence POSCAR element order and verify every TITEL.
- Never display or copy private machine JSON content. It is a path reference.
- Never request credentials or POTCAR content in chat. If configuration is
  missing, instruct the user to run `phonon-agent setup` locally.
- Accept a user-supplied POTCAR ZIP or directory, but never download or
  redistribute it and never choose a variant implicitly.
- `dft-prepare` creates local jobs only. `dft-submit` is a separate, costly,
  explicitly authorized stage.
- Run them separately as `phonon-agent stage <case> dft-prepare` and, only
  after reviewing its summary and receiving authorization,
  `phonon-agent stage <case> dft-submit`.
- Before submission report selected count, recommendation, atom count, total
  tasks, generated cloud job name, machine/MPI core count, POTCAR titles, and
  k-point source. Stop if the declared, machine, and MPI core counts differ.
- Use non-blocking submission by default. Inspect local status first and call
  `phonon-agent resume <run-dir>` to query/collect through DPDispatcher.
- After submission, `phonon-agent stage <case> dft-collect` waits for that
  existing submission, downloads declared results, and converts every
  dpdata-readable OUTCAR to `deepmd/npy`. It never authorizes a pending job.
- Report total, converted, empty and failed OUTCAR counts, total frames, system
  names, and the dpdata path. Empty or unreadable OUTCAR files are skipped; at
  least one converted system is enough to complete the stage.
- Prefer `workflow.yaml` changes. Edit a failed DFT stage script only for a
  concrete technical compatibility issue; never use script changes to bypass
  INCAR, POTCAR, core-count, credential, or submission-authorization checks.
- An OUTCAR parsed by dpdata is a prospective label, not proof of convergence.
  This stage intentionally adds no separate VASP convergence assessment.
- When the user requests the next step, pass the explicit published dpdata path
  to the separate `dpa4-finetuning` skill. Do not treat fine-tuning as another
  phonon-agent stage.
