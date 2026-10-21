# SonicLake

**Benchmarking Workload-Aware Physical Design for Querying Audio Lakes [Experiments & Analysis]**

Anonymous submission to SIGMOD 2027

---

## Overview

SonicLake benchmarks how different physical representations of audio archives affect query
quality and cost across diverse workloads. It treats the choice of audio encoding as a
physical database design decision: raw waveforms (PCM), conventional compression (FLAC, Opus),
neural codec tokens (DAC, EnCodec, Mimi), transcripts, and learned embeddings each preserve
different information at different storage and execution costs.

The system evaluates **four query families** (lexical, speaker/accent, emotion, acoustic event)
across **four datasets** (LibriSpeech, AfriSpeech, CREMA-D, ESC-50) and compares seven layout
strategies: raw-only, smallest-only, average-quality, per-family best, transcript-plus-audio,
workload-aware optimization (SonicLake), and an oracle baseline.

## Repository Structure

```
SonicLake/
├── README.md                   # This file
├── pyproject.toml              # Python package configuration
├── PLAN.md                     # Experiment contract and research questions
├── RUNBOOK.md                  # Execution guide and evaluator requirements
├── BASELINE_RESULTS.md         # Baseline experiment results
├── EXTENSION_RESULTS.md        # Extension experiment results
├── config/
│   ├── paths.env               # Path configuration (edit for your environment)
│   ├── paths.env.example       # Example path configuration
│   ├── experiment_matrix.yaml  # Full experiment matrix definition
│   ├── workload_balanced.json  # Balanced workload mixture
│   ├── workload_sweep.json     # Workload sweep configurations
│   ├── live_serving_regimes.json # Live serving benchmark regimes
│   └── asset_inventory.json    # Dataset and model inventory
├── src/soniclake/              # Core library
│   ├── __init__.py
│   ├── schema.py               # Result schema and validation
│   ├── manifest.py             # Dataset manifest utilities
│   ├── datasets.py             # Dataset loading and configuration
│   ├── io.py                   # I/O utilities
│   ├── systems.py              # Representation system definitions
│   ├── profiles.py             # Quality/cost profile construction
│   ├── evaluation.py           # Evaluation framework
│   ├── optimizer.py            # Workload-aware layout optimizer
│   ├── baselines.py            # Baseline layout strategies
│   ├── pareto.py               # Pareto frontier analysis
│   ├── heldout.py              # Held-out evaluation utilities
│   ├── router.py               # Adaptive layout routing
│   ├── tokenpack.py            # Neural codec token packing
│   └── codec_artifacts.py      # Codec artifact measurement
├── scripts/                    # Experiment scripts
│   ├── build_manifest.py       # Build dataset manifests
│   ├── encode_pcm.py           # PCM encoding
│   ├── encode_flac.py          # FLAC encoding
│   ├── encode_opus.py          # Opus encoding at multiple bitrates
│   ├── encode_dac.py           # DAC neural codec encoding
│   ├── encode_encodec.py       # EnCodec neural codec encoding
│   ├── encode_mimi.py          # Mimi neural codec encoding
│   ├── pack_npz_codes.py       # Pack neural codec tokens
│   ├── evaluate_asr.py         # Lexical query evaluation (ASR)
│   ├── evaluate_keyword_retrieval.py  # Keyword retrieval evaluation
│   ├── evaluate_emotion_prosody.py    # Emotion classification evaluation
│   ├── evaluate_clap_events.py        # Acoustic event evaluation (CLAP)
│   ├── evaluate_policy_baselines.py   # Baseline layout evaluation
│   ├── evaluate_heldout_design.py     # Held-out design evaluation
│   ├── build_production_profiles.py   # Build quality/cost profiles
│   ├── build_retrieval_profiles.py    # Build retrieval profiles
│   ├── aggregate_profiles.py          # Aggregate profiling results
│   ├── optimize_layout.py             # Run layout optimization
│   ├── sweep_layouts.py               # Layout sweep experiments
│   ├── replay_mixed_workloads.py      # Mixed workload replay
│   ├── benchmark_decode.py            # Decode latency benchmarking
│   ├── benchmark_adaptive_layout.py   # Adaptive layout benchmarking
│   ├── benchmark_concurrent_access.py # Concurrent access benchmarking
│   ├── benchmark_live_serving.py      # Live serving benchmarking
│   ├── bootstrap_quality.py           # Bootstrap confidence intervals
│   ├── select_live_regimes.py         # Select live serving regimes
│   ├── make_paper_artifacts.py        # Generate paper tables/figures
│   ├── make_extended_paper_artifacts.py # Extended paper artifacts
│   ├── summarize_extension_results.py # Summarize extension results
│   ├── validate_results.py            # Validate result integrity
│   └── validate_actor_disjoint.py     # Validate actor-disjoint splits
├── tests/                      # Unit and integration tests (24 test files)
│   ├── test_optimizer.py
│   ├── test_baselines.py
│   ├── test_evaluation.py
│   ├── test_pareto.py
│   └── ... (20 more test files)
├── slurm/                      # SLURM batch scripts for cluster execution
│   ├── smoke_matrix.slurm      # Smoke test (20 samples per dataset)
│   ├── learned_codec_smoke.slurm
│   ├── learned_codec_production.slurm
│   ├── asr_smoke.slurm
│   ├── asr_production.slurm
│   └── ... (29 more SLURM scripts)
├── data/                       # Dataset manifests (JSONL)
│   ├── librispeech_test_clean.jsonl
│   ├── librispeech_labeled.jsonl
│   ├── cremad_subset.jsonl
│   ├── cremad_actor_disjoint.jsonl
│   ├── cremad_actor_map.csv
│   └── esc50.jsonl
└── results/
    ├── README.md
    └── smoke/                  # Smoke test outputs (sample results)
        ├── encoding/           # Encoding profiling results
        ├── decode/             # Decode latency results
        ├── evaluation/         # Query evaluation results
        └── representations/    # Encoded representation metadata
```

## Quick Start

### 1. Environment Setup

```bash
git clone <this-repo-url> SonicLake
cd SonicLake

python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,paper]"
```

### 2. Configure Paths

Edit `config/paths.env` to point to your local dataset directories:

```bash
source config/paths.env
export PYTHONPATH=$PWD/src
```

### 3. Download Datasets

The benchmark uses publicly available datasets:

- **LibriSpeech test-clean**: https://www.openslr.org/12
- **CREMA-D**: https://github.com/CheyneyComputerScience/CREMA-D
- **ESC-50**: https://github.com/karolpiczak/ESC-50
- **AfriSpeech**: https://huggingface.co/datasets/intronhealth/afrispeech-dialog

Build manifests:

```bash
python scripts/build_manifest.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Encode Representations

```bash
# Conventional
python scripts/encode_pcm.py
python scripts/encode_flac.py
python scripts/encode_opus.py --bitrate 24
python scripts/encode_opus.py --bitrate 48
python scripts/encode_opus.py --bitrate 96

# Neural codecs
python scripts/encode_dac.py --codebooks 4
python scripts/encode_encodec.py --codebooks 3
python scripts/encode_mimi.py --codebooks 8
# ... (see slurm/ for full configurations)
```

### 6. Evaluate Query Families

```bash
# Lexical (ASR-based keyword retrieval)
python scripts/evaluate_asr.py
python scripts/evaluate_keyword_retrieval.py

# Emotion (CREMA-D)
python scripts/evaluate_emotion_prosody.py

# Acoustic event (ESC-50 via CLAP)
python scripts/evaluate_clap_events.py

# Build profiles and optimize
python scripts/build_production_profiles.py
python scripts/optimize_layout.py
python scripts/evaluate_policy_baselines.py
```

### 7. Statistical Analysis

```bash
python scripts/bootstrap_quality.py
python scripts/sweep_layouts.py
python scripts/evaluate_heldout_design.py
python scripts/make_paper_artifacts.py
```

## Physical Representations

| Family | Configurations |
|---|---|
| Lossless | PCM (WAV s16le); FLAC (compression level 8) |
| Conventional lossy | Opus at 24, 48, 96 kbps |
| Neural acoustic | DAC (4/8/16 codebooks); EnCodec (3/6/12 codebooks) |
| Neural semantic/hybrid | Mimi (8/16/32 codebooks) |
| Derived text | ASR transcript + inverted index |
| Derived vectors | Speaker, emotion, and acoustic-event embeddings |

## Layout Baselines

1. **Raw-only**: retain PCM, answer every query from the original signal
2. **Smallest-only**: choose the minimum-storage representation
3. **Average-quality**: single representation with best unweighted mean quality
4. **Per-family best**: independently best representation per query family
5. **Transcript-plus-audio**: transcript index + one compressed audio
6. **SonicLake**: workload-aware optimizer selecting complementary representations
7. **Oracle**: optimize using held-out true quality and cost

## Metrics

- **Storage**: bytes per audio-hour
- **Encoding**: one-time construction time (seconds)
- **Decoding**: per-query decode latency (ms)
- **Lexical**: keyword recall, F1
- **Speaker/Accent**: classification accuracy, retrieval accuracy
- **Emotion**: six-way classification accuracy, macro-F1 (CREMA-D)
- **Acoustic event**: ESC-50 classification accuracy
- **Statistical**: bootstrap 95% CIs, Pareto frontier analysis

## Hardware Requirements

- GPU: NVIDIA A100 80GB (for neural codec encoding) or L4 (for evaluation)
- The full benchmark takes approximately 24 GPU-hours on A100
- Individual experiments can run on consumer GPUs with reduced batch sizes

## License

This benchmark artifact is provided for research purposes. The underlying datasets
(LibriSpeech, CREMA-D, ESC-50, AfriSpeech) are subject to their respective licenses.
See each dataset's documentation for terms.
