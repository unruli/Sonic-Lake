# FLAC and keyword-retrieval extension

## Completed jobs

- FLAC encoding: `45918912`
- FLAC task evaluation: `45918913`
- FLAC decode benchmark: `45918914`
- FLAC profile and optimizer analysis: `45919237`
- Keyword retrieval across source plus 14 candidates: `45921559`
- Retrieval-aware profiles and optimizer sweep: `45921560`
- FLAC fixed-policy comparison: `45923894`
- FLAC-aware five-seed robustness: `45924163`

## FLAC result

FLAC completed all 4,920 encodings and downstream evaluations without failure. It exactly reproduces source
quality: WER 2.683%, event accuracy 93.05%, and emotion macro-F1 0.48446. Its aggregated footprint is 91.94
MB/audio-hour versus 182.04 MB/audio-hour for PCM. LibriSpeech decode RTF is 0.000400.

FLAC is the strongest fixed-format held-out baseline: 960 valid regimes, 1.97% median oracle regret, and 8.47%
p95 regret. PCM has 924 valid regimes and 55.96% p95 regret after the held-out oracle gains access to FLAC.

## Keyword retrieval result

The workload selects 500 terms deterministically from 2,620 LibriSpeech documents, builds an inverted index over
each representation's frozen ASR hypotheses, and scores against reference-derived relevance sets. Macro-F1 spans
0.9690--0.9784. Index storage is approximately 0.14 MB/audio-hour and median in-memory posting lookup is
0.086--0.116 microseconds. ASR construction dominates at 274--302 seconds/audio-hour.

With transcript indexes attached to audio profiles, both single and adaptive layouts are feasible in 104 of 168
regimes. Adaptive improves all 104, with 5.64% median, 48.54% p95, and 51.70% maximum latency reduction. Opus-24
serves lexical queries in 96 adaptive layouts and Mimi-32 in eight.

## Final robustness and economics

Across five seeds and 10--50% calibration, uncertainty-aware selection covers 2,320 of 3,128 point-feasible
regimes (74.2%). On the matched set it raises held-out quality satisfaction from 92.9% to 97.9%, with 6.3% p95
latency regret. This is risk reduction, not a guarantee: the final candidate set exposes residual violations and
4.4% held-out budget exceedance.

Point-adaptive layouts beat workload-aware single layouts in 609 of 1,102 jointly valid regimes. Median
construction break-even among valid wins is 9,484 queries/audio-hour; robust adaptive requires 13,217. The
deployment claim therefore remains restricted to hot archives.

## Mixed-query router replay

The catalog-driven router was executed over every feasible retrieval-aware design: 208 catalogs, 10,000 seeded
mixed queries per catalog, and 2.08 million total dispatches. The control path performs catalog lookup, family
dispatch, and deterministic record selection; execution latency is replayed from per-record decode/evaluator
measurements, with lexical requests using the measured 500-term posting-list lookup timings. Median routing
overhead across catalogs was 1.3 microseconds, and the p95 across catalog-level p95 values was 3.0 microseconds.
This validates the router implementation but is not a live model-serving benchmark; concurrency, queueing, and
storage contention remain outside the experiment.

## Local artifacts

- `results/production/analysis/soniclake_flac_retrieval_jobs45918912-45921560.tar.gz`
- `results/production/analysis/flac_policy_job45923894.tar.gz`
- `results/production/analysis/flac_robust_multiseed_job45924163.tar.gz`
- `results/production/flac_analysis/extension_summary.json`
- `results/production/paper_artifacts_flac/`
- `results/production/router/mixed_workload_replay.json`
