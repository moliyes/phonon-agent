# Workflow contract

The executable stages are `phonon-check`, `md`, `entropy`, `selection`,
`dft-prepare`, `dft-submit`, and `dft-collect`. Execute them individually with
`phonon-agent stage` and inspect each result before continuing. A run
directory is immutable evidence: do not edit `config.resolved.yaml`,
`state.json`, completed attempts, or script snapshots by hand.

New schema-version-2 cases contain full editable implementations in `scripts/`.
Change configuration first. A failed stage may be retried after a necessary
technical script fix; changing physical methods, selection logic, or safety
limits requires user approval. Changing inputs of a completed stage requires a
new run, while future-only settings such as `selection.count` may be adjusted
before that stage runs.

`dft-collect` may automatically wait for and recover an existing submitted
DPDispatcher job, but it cannot create an unapproved submission. It converts
all dpdata-readable OUTCAR files into `deepmd/npy`, reports empty and failed
inputs, and completes when at least one system was exported. No additional
OUTCAR completeness or convergence policy is imposed.

The initial phonon check pauses below `-0.1 THz`. Entropy selection recommends
the first scanned count whose raw entropy is at least 95% of the maximum raw
entropy in that scan. A user-provided `selection.count` overrides the
recommendation but both values remain in the summary.

No VASP-task cap is imposed. Reporting task scale before submission is
mandatory. Machine and resources JSON are private references and must not be
printed, copied, summarized field-by-field, or committed.

User-level private defaults live under `~/.config/phonon-agent/`. If they are
missing at `dft-prepare` or `dft-submit`, instruct the user to run
`phonon-agent setup`; never collect secrets in chat. Validation must use the
requested `--until` stage so local phonon/MD/entropy work is not blocked by
future VASP inputs. POTCAR may be a user-supplied ZIP or directory, but element
variant mappings remain explicit per project.
