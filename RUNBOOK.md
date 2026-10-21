# SonicLake runbook

## What is executable now

- Immutable, checksummed JSONL manifests for WAV and FLAC sources.
- Opus encoding with artifact size and wall-clock construction measurements.
- A normalized sample-level result contract with strict validation.
- Aggregation from sample rows to representation/query-family profiles.
- Exact physical-design subset optimization for the frozen, small representation roster.
- Pareto-frontier analysis utilities.
- HPC cluster smoke-job scaffold with verified dataset and environment paths.

## Immediate commands on Blue

```bash
cd ${SL_ROOT:-.}
export PYTHONPATH=$PWD/src
${CODEC_EVAL_ROOT}/.venv/bin/python -m unittest discover -s tests -v
sbatch slurm/smoke_matrix.slurm
```

Do not submit the production matrix until the 20-sample smoke job has succeeded for every retained adapter and
its output passes `scripts/validate_results.py`.

## Adapter contract

Every codec arm must preserve two artifacts:

1. The actual stored representation used in the storage calculation (for example entropy-coded payload or a
   deterministic packed token stream), not merely a reconstructed WAV or a Python object dump.
2. A decoded waveform used by downstream evaluators.

Every evaluator emits `RunResult` rows with the exact model revision and environment. Quality is normalized to
`[0,1]`, with the task's native metric also retained in evaluator-specific diagnostics.

## Required evaluator freezes

- Lexical: one ASR model plus deterministic keyword workload and source transcripts.
- Speaker: one frozen speaker embedding model; retrieval protocol fixed before production runs.
- Accent: existing validated classifier, with a source-audio ceiling reported.
- Emotion: one frozen classifier evaluated against CREMA-D source labels, with source-audio ceiling.
- Acoustic event: one frozen ESC-50 classifier or CLAP zero-shot protocol, fixed prompts, and source-audio ceiling.

The source-audio ceiling is mandatory. If an evaluator is weak on original audio, codec differences are not
interpretable for that family.

## Blocking work before production

1. Download ESC-50 and record its release/checksum/license.
2. Decide whether the 300-file CREMA-D slice is sufficiently balanced; otherwise acquire the complete release.
3. Implement deterministic packed-token storage for DAC, EnCodec, and Mimi. Existing reconstruction scripts write
   WAV files and cannot support a defensible codec-storage comparison by themselves.
4. Add codec-specific decode adapters that consume exactly the stored payload.
5. Freeze evaluator revisions, prompts, label maps, and source-audio ceiling tests.
6. Generate balanced calibration and held-out manifests without speaker/item leakage.
7. Run the complete smoke matrix and remove unstable arms.

## Minimum evidence for the paper

- At least 200 held-out recordings per applicable family where the public dataset permits it; report smaller
  subsets transparently rather than padding with correlated segments.
- PCM, Opus at three bitrates, and at least three neural-codec operating points spanning acoustic and
  semantic/hybrid representations.
- Source-audio evaluator ceiling and confidence intervals for every query family.
- Measured stored bytes, encode time, decode time, and query time.
- Balanced and skewed workload mixtures, storage budgets, and quality thresholds.
- Calibration-versus-held-out optimizer evaluation and one workload-shift stress test.
- Baselines and ablations listed in `PLAN.md`.

## Submission gate

The paper is not ready until all headline claims can be regenerated from immutable raw rows, every selected layout
is learned from calibration data and evaluated on held-out data, and no result table depends on an artifact already
used as evidence in another submission.
