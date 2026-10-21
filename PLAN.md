# SonicLake execution plan

## Research questions

1. How do candidate physical representations trade storage and decoding cost against query quality?
2. Is representation loss attribute-selective across lexical, speaker/accent, emotion, and acoustic-event queries?
3. When is one representation sufficient, and when is materializing complementary representations worthwhile?
4. Can a workload-aware selector meet per-family quality constraints at lower total cost than fixed layouts?
5. How robust is the selected layout under workload and quality-estimation shift?

## Physical representations

| Family | Initial configurations |
|---|---|
| Lossless | PCM/WAV; FLAC at compression level 8 |
| Conventional | Opus at low, medium, and high bitrate |
| Neural acoustic | DAC and EnCodec at up to three codebook depths |
| Neural semantic/hybrid | Mimi plus one existing runnable semantic codec if stable |
| Derived text | ASR transcript plus inverted index |
| Derived vectors | speaker/accent, emotion, and acoustic-event embeddings |

The final roster must be frozen after a smoke test. An unstable codec is removed rather than repaired during the
main experiment.

## Datasets and query workloads

| Dataset | Public task | Query-family metric |
|---|---|---|
| LibriSpeech | speech content and speaker labels | keyword recall/F1; speaker retrieval |
| AfriSpeech subset | accented speech | keyword recall/F1; accent classification |
| CREMA-D | acted emotion | emotion classification/query F1 |
| ESC-50 | environmental sound events | event classification/query F1 |

All samples receive immutable IDs. Train/test partitions follow the source datasets. Pretrained evaluators are
frozen before evaluating representations.

## Cost model

For materialized representation set M and workload W:

```text
minimize  storage(M) + sum_q frequency(q) * execution_cost(q, M)
subject to predicted_quality(q, M) >= threshold(q) for every query family q
```

Report three cost views separately rather than collapsing them prematurely:

- storage bytes per audio-hour;
- one-time encoding/index-construction time;
- per-query decoding and execution latency.

Dollar conversion, if included, is explicitly illustrative and derived from measured runtime.

## Baselines

1. Raw-only: retain PCM and answer every query from the original signal.
2. Smallest-only: choose the minimum-storage representation.
3. Average-quality: choose the single representation with best unweighted mean quality.
4. Per-family best: materialize the independently best representation for every family.
5. Transcript-plus-audio: conventional transcript index plus one compressed audio representation.
6. SonicLake: optimize the representation set for workload and constraints.
7. Oracle: optimize using held-out true quality and cost.

## Required analyses

- Quality-versus-storage and quality-versus-latency Pareto frontiers.
- Attribute-retention heat map across representations and operating points.
- Layout selected as workload mixture changes.
- Sensitivity to storage budget and quality thresholds.
- Calibration error between profiled and held-out quality.
- Ablation of multi-representation materialization.
- At least one workload-shift stress test.
- Bootstrap confidence intervals over recordings.

## Guardrails

- Do not reuse an experimental table or central claim from another submitted paper.
- Treat existing codec infrastructure as tooling, not as previously unpublished evidence automatically available here.
- Do not describe task-model output as ground truth when human/source labels are available.
- Separate measured costs from modeled costs.
- Do not claim universal codec rankings beyond the evaluated datasets, tasks, and operating points.

## Seven-day schedule

### Day 0: submission registration

- Submit title, abstract, authors, and COIs.
- Freeze datasets, codec roster, query metrics, and paper claims.
- Inventory reusable code and dataset locations.

### Day 1: manifest and smoke matrix

- Build balanced dataset manifests.
- Run all representation/task combinations on 20 samples per dataset.
- Remove broken configurations and freeze environments.

### Days 2-3: production profiling

- Encode production subsets.
- Run frozen query evaluators.
- Record storage, construction time, decode time, and query latency.

### Day 4: optimizer and workload experiments

- Fit quality/cost profiles using calibration partitions.
- Run layout baselines, budget sweeps, and workload-mixture sweeps.

### Day 5: statistical analysis and figures

- Bootstrap intervals, significance tests, sensitivity analysis, and error inspection.
- Freeze generated tables and figures.

### Day 6: manuscript completion

- Complete methods, results, related work, limitations, and reproducibility material.

### Day 7: audit and submission

- Reproduce headline tables from raw outputs.
- Inspect anonymity, formatting, citations, PDF layout, and claim/evidence alignment.
