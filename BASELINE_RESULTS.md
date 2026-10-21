# Policy baselines and break-even results

## Run

- Original 13-candidate Slurm job: `45917776`
- Final FLAC-inclusive Slurm job: `45923894`
- Seeds: 2027--2031
- Calibration fractions: 10%, 20%, 50%
- Conditions per policy: 2,520
- Policy rows: 25,200
- Break-even rows: 2,648

## Main policy comparison

| Policy | Selected | Held-out valid | Quality success when selected | Median valid regret | p95 valid regret |
|---|---:|---:|---:|---:|---:|
| PCM | 936 | 924 | 98.7% | 11.44% | 55.96% |
| FLAC | 1,044 | 960 | 96.6% | 1.97% | 8.47% |
| Opus-48 | 660 | 560 | 84.8% | 10.20% | 60.42% |
| Smallest | 56 | 0 | 0.0% | — | — |
| Random feasible single | 1,512 | 1,095 | 74.5% | 8.40% | 57.99% |
| Workload-agnostic single | 1,512 | 1,188 | 81.7% | 2.10% | 11.80% |
| Workload-aware single | 1,512 | 1,173 | 80.8% | 2.11% | 11.80% |
| Point adaptive | 1,616 | 1,242 | 80.7% | 1.73% | 7.49% |
| Uncertainty-aware adaptive | 1,184 | 1,102 | 98.0% | 1.85% | 7.91% |
| Held-out adaptive oracle | 1,500 | 1,500 | 100.0% | 0.00% | 0.00% |

Regret is measured against the held-out adaptive oracle only where the policy is held-out valid.

## Matched adaptive-versus-single comparison

Point adaptive and workload-aware single are both held-out valid in 1,102 regimes. Point adaptive is faster in
609 (55.3%), with 1.44% median, 9.60% p95, and 34.01% maximum latency reduction among wins. Point adaptive is
uniquely valid in 140 regimes; single is uniquely valid in 71.

Robust adaptive and workload-aware single are both held-out valid in 1,002 regimes. Robust adaptive is faster in
513 (51.2%), with 1.24% median, 9.11% p95, and 31.25% maximum latency reduction among wins. Robust adaptive is
uniquely valid in 100 regimes; single is uniquely valid in 171.

## Construction break-even

| Candidate over workload-aware single | Valid comparisons | Faster | Median extra build s/audio-hour | Median extra MB/audio-hour | Median break-even queries/audio-hour | p95 break-even |
|---|---:|---:|---:|---:|---:|---:|
| Point adaptive | 1,167 | 627 | 29.79 | 22.82 | 9,484 | 56,220 |
| Robust adaptive | 1,060 | 527 | 29.79 | 92.31 | 13,217 | 84,621 |

Only 1.6% of point-adaptive wins and 0.2% of robust-adaptive wins break even by 1,000 queries per archived
audio-hour. None break even by 100 queries. Adaptive materialization is therefore an optimization for hot audio,
not a default for cold archives.

## Artifacts

- `results/production/flac_analysis/policy_baselines_with_flac.jsonl`
- `results/production/flac_analysis/break_even_with_flac.jsonl`
- `results/production/flac_analysis/policy_baselines_with_flac_summary.json`
- `results/production/analysis/flac_policy_job45923894.tar.gz`
