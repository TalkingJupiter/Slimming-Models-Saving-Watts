#!/usr/bin/env bash
#SBATCH --job-name=kd_submitter_single_node
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=00:15:00
#SBATCH --output=logs/submission/%x_%j.out
#SBATCH --error=logs/submission/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"


RB=$(sbatch --parsable scripts/run_response_based_single_node.sh)
FB=$(sbatch --parsable scripts/run_feature_based_single_node.sh)
RELB=$(sbatch --parsable scripts/run_relation_based_single_node.sh)

echo "[SUBMITTED] Response-Based=$RB  Feature-Based=$FB  Relation-Based=$RELB"

harness=$(sbatch --parsable --dependency="afterok:${RB}:${FB}:${RELB}" eval/hardness_submitter.sh)
echo "[SUBMITTED] Eval-Submitter=$harness (afterok:$RB:$FB:$RELB)"
