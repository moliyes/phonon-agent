# phonon-agent

`phonon-agent` is a resumable materials workflow for Hermes Agent, connecting
DPA4 sampling, VASP labeling, and local model fine-tuning:

```text
Initial DPA4 phonon check -> NVT/NPT molecular dynamics
-> QUESTS entropy scan and frame selection -> VASP static labeling inputs
-> Bohrium/DPDispatcher submission and collection
-> OUTCAR conversion to DeepMD NPY data -> separate local DPA4 fine-tuning skill
```

It uses [dpa-phonon](https://github.com/moliyes/dpa-phonon) for phonon checks
without modifying that project's source. Remote jobs require explicit user
authorization.

## Installation

Use Python 3.11+ in an environment with a compatible DeepMD-kit/PyTorch runtime
already installed. Activate that environment, then run:

```bash
git clone https://github.com/moliyes/phonon-agent.git
cd phonon-agent
python -m pip install -e ".[sampling,conversion,test]"
python -m pip install dpa-phonon
phonon-agent --version
ph --version
```

If that environment contains the older `phonon-kit` distribution, uninstall it
before installing `dpa-phonon`: they share the `phonon_kit` import and `ph`
command. DeepMD-kit, VASP, and licensed POTCAR files are user-provided.

The repository includes Hermes skills for
[workflow orchestration](skills/materials-science/phonon-agent/SKILL.md),
[MD sampling](skills/materials-science/dpa4-md-sampling/SKILL.md),
[VASP labeling](skills/materials-science/vasp-labeling/SKILL.md), and
[DPA4 fine-tuning](skills/materials-science/dpa4-finetuning/SKILL.md).

## Initial private configuration

Before the first phonon, MD, or VASP workflow, run:

```bash
phonon-agent setup
phonon-agent setup --check
```

Setup asks for an existing DeepMD-kit environment directory or its Python
executable. After checking DeepMD, it installs missing workflow tools into that
environment and creates `~/.local/bin/phonon-agent` and `~/.local/bin/ph`.
It does not install or upgrade DeepMD-kit. Ensure `~/.local/bin` is on your
`PATH`, or use those commands by their full paths.

You can then run `ph model freeze` in the configured environment to create a
local DPA4 `.pt2` compatible with that machine's DeepMD/PyTorch/CUDA stack.
Without a local frozen model, `dpa-phonon` falls back to its packaged `.pt`
checkpoint.

Setup collects passwords or access keys through hidden terminal prompts and
creates configuration files with restricted permissions:

```text
~/.config/phonon-agent/
├── config.yaml
└── bohrium/
    ├── machine.json
    └── resources.json
```

Do not send passwords, access keys, or POTCAR contents through Hermes chat or
commit them to Git. Setup does not migrate credentials from older projects;
rotate any credentials that have been exposed. Supply POTCAR files from your
own licensed source, either as a ZIP archive or a directory containing
`<family>/<variant>/POTCAR`.

User configuration supplies the DeepMD Python executable, the `ph` command,
POTCAR location, and 16-core Bohrium defaults. Individual projects can override
these values in `workflow.yaml`. Confirm the element-to-POTCAR-variant mapping
for each project. See the [private configuration guide](docs/private-configuration.md)
(in Chinese) for details.

## Usage

```bash
phonon-agent init my-case
# Creates workflow.yaml, input placeholders, and seven editable stage scripts.
# Replace inputs/POSCAR, INCAR, and KPOINTS; configure workflow.yaml first.
phonon-agent validate my-case --until dft-prepare
phonon-agent plan my-case
phonon-agent stage my-case phonon-check
phonon-agent status my-case
phonon-agent stage my-case md
phonon-agent status my-case
phonon-agent stage my-case entropy
phonon-agent status my-case
phonon-agent stage my-case selection
phonon-agent status my-case
phonon-agent stage my-case dft-prepare
phonon-agent status my-case
# Submit only after explicit user authorization.
phonon-agent stage my-case dft-submit
# Wait for existing jobs, collect outputs, and convert the results.
phonon-agent stage my-case dft-collect
phonon-agent status my-case
```

The workspace's `scripts/` directory contains the implementations for the
phonon check, MD, entropy scan, frame selection, DFT preparation, submission,
and collection/conversion. Each `stage` command runs one script and records a
read-only script snapshot, SHA256 hash, logs, and a separate attempt directory.
After a failure, you can fix the current stage script and retry. Changing a
workspace script does not silently rerun completed stages.

Normally, edit only `workflow.yaml`. Change a script only when configuration
cannot address a specific technical compatibility issue. Obtain user approval
before changing physical methods, ensemble behavior, QUESTS algorithms, or
safety thresholds. Do not execute stage scripts directly or edit
`runs/*/state.json`, saved configuration snapshots, or historical script snapshots.

Validation follows the requested stage. Phonon, MD, and QUESTS stages do not
require Bohrium or POTCAR. `dft-prepare` checks POTCAR inputs; `dft-submit` also
checks Bohrium credentials and the 16-core command. Remote submission occurs
only through the explicitly requested `dft-submit` stage.

`dft-collect` does not create new submissions. It waits for the existing
DPDispatcher jobs, downloads their outputs, and converts OUTCAR files readable
by dpdata into DeepMD NPY datasets under the run directory:

```text
stages/60-dft-collect/current/dpdata/
```

Empty or unparseable OUTCAR files are skipped and recorded in `summary.json`.
The stage completes when at least one data system is converted successfully.
This does not imply that every VASP job is valid, and collection does not
independently verify VASP convergence.

## Local DPA4 fine-tuning

Fine-tuning is separate from the seven-stage workflow. Hermes uses the
`dpa4-finetuning` skill and passes the dataset path published by `dft-collect`
to its bundled script. From this repository's root:

```bash
python skills/materials-science/dpa4-finetuning/scripts/run_finetuning.py \
  --data my-case/runs/<run-id>/stages/60-dft-collect/current/dpdata \
  --output my-case/finetuning/<run-id>-001
```

Replace `<run-id>` with the actual run identifier before executing the command.
The script defaults to the `dpa4_0704.pt` checkpoint packaged with `dpa-phonon`,
copies the model and data into a self-contained directory, sets training steps
to the total number of frames multiplied by 30, and runs local training,
freezing, and training-set evaluation in that order.

It rejects `.pt2` files as fine-tuning inputs, does not generate a Bohrium
`job.json`, and refuses to overwrite an existing output directory. Add
`--prepare-only` to prepare the directory without starting training.
Training-set metrics are diagnostics, not evidence of generalization.

## Tests

In the configured development environment:

```bash
python -m pytest
```
