---
name: dpa4-finetuning
description: Fine-tune DPA4 locally from explicit DeepMD NPY data.
license: MIT
metadata:
  hermes:
    version: 0.1.0
    author: moliyes, Hermes Agent
    platforms: [linux]
    tags: [materials-science, DPA4, DeepMD, fine-tuning]
    category: materials-science
    related_skills: [phonon-agent, vasp-labeling, phonon-kit]
    requires_tools: [terminal]
---

# DPA4 Fine-tuning

Use this skill after DFT labels have been converted to DeepMD NPY data. It
creates a self-contained local workspace, fine-tunes a DPA4 checkpoint, freezes
the result for inference, and evaluates that frozen model on the training data.
This is not a phonon-agent stage and does not submit Bohrium jobs.

## Required inputs

- Require an explicit DeepMD NPY data path. For a phonon-agent run, suggest
  `runs/<run-id>/stages/60-dft-collect/current/dpdata`, but never discover or
  choose a run implicitly.
- Use the packaged `dpa4_0704.pt` by default. Accept a user-specified DPA4
  `.pt` checkpoint, but never use a `.pt2` model for fine-tuning.
- Use total labeled frames multiplied by 30 for `numb_steps`. Pass
  `--numb-steps` only when the user explicitly requests another positive
  value.

## Procedure

1. Inspect the data path and the preceding dft-collect summary when available.
   Partial conversion is usable, but report that it does not represent every
   submitted VASP task.
2. Choose a new output path without overwriting anything. For standard cases
   use `<case>/finetuning/<run-id>-001`; otherwise use
   `<data-parent>/<data-name>-finetune-001`. Increment the suffix until unused.
3. Before starting, report the data path, systems, total frames, base `.pt`,
   calculated steps, configured DeepMD environment, and output path.
4. Run the bundled executable:

   ```bash
   scripts/run_finetuning.py --data DATA --output OUTPUT
   ```

   Add `--model MODEL.pt` only for an explicit model and `--prepare-only`
   when the user wants inspection without training. Run normal training in the
   background because it may take a long time. Do not start a second process
   for the same output directory.
5. The script copies the data and model, then runs train -> freeze -> test. Do
   not edit the generated `input.json` or scientific settings unless the user
   explicitly asks. If a command fails, preserve the workspace, report the
   failed step and log tail, and wait for direction; do not auto-restart,
   delete, or disguise the failure.
6. On success, report `frozen.pt2`, `lcurve.out`, `infer_train_log`, and
   `pred_train.*`. Summarize metrics as training-set diagnostics only; they
   do not establish generalization or DFT-level accuracy.

The script reads the DeepMD Python configured by `phonon-agent setup` and
uses the adjacent `dp` executable. If unavailable, instruct the user to
configure or repair that environment; never install or upgrade DeepMD-kit.
