#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${SL_ROOT:-.}
cd "$PROJECT_ROOT"
mkdir -p logs results/actor_disjoint/evaluation

prep_submission=$(sbatch --parsable slurm/prepare_cremad_actor_disjoint.slurm)
prep_job=${prep_submission%%;*}
evaluation_submission=$(sbatch --parsable --dependency="afterok:${prep_job}" slurm/emotion_actor_disjoint.slurm)
evaluation_job=${evaluation_submission%%;*}

printf 'preparation_job=%s\nevaluation_array=%s\ndependency=afterok:%s\n' \
  "$prep_job" "$evaluation_job" "$prep_job"
