# Held-out and uncertainty-aware selection results

## Completed experiments

- Baseline held-out job: `45865156`
- Initial robust comparison: `45913366`
- Confidence/sample-floor sensitivity: `45913573`
- Five-seed run and retries: `45913980`, `45914258`, `45914396`
- Split seeds: 2027--2031
- Calibration fractions: 1%, 2%, 5%, 10%, 20%, 50%
- Holdout: disjoint 50% hash partition
- Recommended robust policy: paired one-sided jackknife bound at 90% confidence, with at least 10 records per
  query family

## Main five-seed result

Across all calibration fractions, the point selector chose 6,460 layouts and satisfied all held-out quality
constraints in 59.2% of them. The robust selector chose 3,032 layouts and satisfied all constraints in 91.9%.
On the 3,032 configurations selected by both methods, point-estimate satisfaction was 84.5% and robust
satisfaction was 91.9%. Robust selection covered 46.9% of point-feasible configurations because it correctly
abstained on the 1--2% splits and rejected additional uncertain cases.

The 5% fraction remains underpowered: robust held-out satisfaction was 71.6%. At 10%, 20%, and 50% calibration,
the robust selector made 2,168 selections with zero observed held-out violations across five seeds. It covered
73.7% of the 2,940 point-feasible configurations. On this matched set:

- robust held-out constraint satisfaction: 100.0%
- point held-out constraint satisfaction: 93.6%
- robust median / p95 latency regret: 0.0% / 5.0%
- point median / p95 latency regret: 0.0% / 4.1%
- robust held-out storage-budget exceedance: 0.0%

The evidence therefore supports a practical 10% calibration floor for this dataset roster. The zero violations
are empirical over five splits, not a formal distribution-free guarantee.

## Sensitivity result on seed 2027

| Confidence | Minimum samples | Robust selections | Held-out satisfaction | Point-feasible coverage | Matched p95 regret |
|---:|---:|---:|---:|---:|---:|
| 80% | 10 | 640 | 98.9% | 51.6% | 2.8% |
| 90% | 10 | 592 | 100.0% | 47.7% | 2.5% |
| 95% | 10 | 576 | 100.0% | 46.5% | 2.5% |
| 90% | 20 | 448 | 100.0% | 36.1% | 2.8% |
| 95% | 30 | 288 | 100.0% | 23.2% | 3.7% |

The 90% / 10-sample setting is the best observed safety--coverage compromise and is the setting used by the
five-seed experiment.

## Local result artifacts

- `results/production/analysis/robust_multiseed_summary.json`
- `results/production/analysis/robust_sensitivity_summary.json`
- `results/production/analysis/robust_heldout_design.jsonl`
- `results/production/analysis/robust_workload_shift.jsonl`
- `results/production/analysis/robust_multiseed/`

## Next evidence priorities

1. ~~Add explicit policy baselines and the held-out oracle.~~ Completed in job `45917776`.
2. ~~Compute construction/storage break-even query counts for adaptive wins.~~ Completed in job `45917776`.
3. ~~Add FLAC and a transcript-plus-Opus retrieval baseline.~~ Completed in jobs `45918912`--`45919237` and
   retrieval jobs `45921559`--`45921560`.
4. ~~Build the keyword-retrieval workload.~~ Completed for 500 deterministic terms and all 14 candidates in job
   `45921559`; full mixed-query router execution remains open.
