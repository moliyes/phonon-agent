---
name: phonon-agent
description: Orchestrate DPA4 sampling, VASP labeling, and fine-tuning.
version: 0.6.0
author: moliyes, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [materials-science, phonons, DPA4, VASP]
    category: materials-science
    related_skills: [dpa4-md-sampling, vasp-labeling, dpa4-finetuning, phonon-kit]
    requires_tools: [terminal]
---

# Phonon Agent

Use this skill when a user asks to sample DPA4 structures, select training
frames, prepare or submit VASP labels, fine-tune a model, or compare phonon
properties. Use `phonon-kit` for the property calculations themselves.
The workflow purpose is DPA4 finite-temperature exploration -> QUESTS
prescreening and frame selection -> true VASP labels -> later fine-tuning and
property comparison.

## Rules

- Verify files and state with tools. Never invent filenames, command output,
  calculation completion, model quality, or scientific results.
- If no case exists, run `phonon-agent init <case>`; use `--profile agi`
  only at the user's explicit request. Replace the placeholder POSCAR and
  inspect `workflow.yaml` plus `scripts/README.md`.
- In an existing case, inspect `workflow.yaml`, the seven editable scripts, and
  the latest `state.json` before changing or running anything.
- Do not modify the phonon-kit checkout.
- Do not relax a structure unless the user chooses that response after the
  initial phonon check pauses for an imaginary mode.
- Preserve user INCAR settings. A conflict is a reason to stop, not permission
  to repair it.
- A request to prepare jobs is not authority to submit them. Execute the
  `dft-submit` stage only when submission is explicit.
- Before submission, report selected structures, atoms per structure, total
  jobs, generated cloud job name, machine/MPI core count, POTCAR mapping, and
  whether KPOINTS or KSPACING is used.
- Never ask for or display a password, access key, ticket, or POTCAR content in
  chat. If private configuration is missing, tell the user to run
  `phonon-agent setup` in their terminal and wait for confirmation.
- POTCAR must be supplied legally by the user as a ZIP or directory. Never
  download it or choose element variants for the user.
- DeepMD-kit is user-managed. If its configured Python is absent or cannot
  import DeepMD, instruct the user to install it and rerun `phonon-agent setup`;
  never install or upgrade DeepMD automatically. Use the globally configured
  `ph` command and configured DeepMD environment for all compute stages.
- Before expensive `builtin:dpa4` calculations, prefer `ph model freeze` in the
  configured environment. If no machine-local `.pt2` exists, explicitly report
  that phonon-kit will fall back to its packaged `.pt` checkpoint.
- Run MD, phonon calculations, and remote operations in the background with
  completion notification. Do not start a second process for the same run.
- After explicit submission, use `phonon-agent stage <case> dft-collect`.
  It may block while recovering the existing DPDispatcher submission, so run
  it in the background. It must never turn a pending submission into a new
  remote job.
- Fine-tuning is a separate local skill, not an eighth workflow stage. After
  dft-collect, use `dpa4-finetuning` with the published dpdata path; never
  pass a frozen `.pt2` as its base model.
- Prefer configuration changes. Edit only the current failed stage script when
  a concrete technical problem cannot be solved in `workflow.yaml`. Explain
  the fix before retrying. Ask first before changing physics, ensemble meaning,
  QUESTS logic, or any safety threshold. Never edit state, resolved config,
  completed outputs, or historical script snapshots.

## Procedure

1. Run `phonon-agent validate <case> --until <stage>` and
   `phonon-agent plan <case>`.
2. Execute exactly one authorized stage using
   `phonon-agent stage <case> <stage>`, beginning with `phonon-check`.
3. Inspect `phonon-agent status <case-or-run-dir>`, the attempt log, and the
   stage result before choosing or configuring the next stage.
4. If status is `waiting_user`, quote its reason and stop. Do not bypass it.
5. Use `phonon-agent resume <run-dir>` only to retry the current failed or
   waiting stage, or refresh a submitted DFT stage.
6. After dft-collect, report total OUTCAR files, converted, empty and failed
   counts, total frames, system names, and the published dpdata path. Partial
   conversion is usable but is not evidence that every VASP task succeeded.
7. If the user requests fine-tuning, hand the explicit published dpdata path to
   the `dpa4-finetuning` skill and keep the long local run in the background.
8. Report concrete result and state paths. Distinguish workflow completion from
   proof of DFT accuracy.

Read `references/workflow.md` before creating or changing a workflow config.
