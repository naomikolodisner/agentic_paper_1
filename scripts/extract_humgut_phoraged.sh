#!/usr/bin/env bash
# =============================================================================
# extract_humgut_phoraged.sh
#
# One array task per shard (7 total, HumGut_phoraged_1-5k.tar.gz through
# HumGut_phoraged_30k-31k.tar.gz) -- each task streams through its own shard
# once (tarfile stream mode, single sequential decompression pass, avoids
# double-reading the archive) and extracts whichever of the ~3455 needed
# genomes (config.OUT_REPRESENTATIVE_GENOMES) happen to be in it. All tasks
# write into the same shared config.HUMGUT_PROPHAGE_REMOVED_DIR -- safe since
# each genome only exists in exactly one shard, so there's no write
# collision between tasks.
#
# Idempotent -- scripts/extract_humgut_phoraged.py skips any genome whose
# output file already exists.
#
# Usage:
#   sbatch --array=0-6 scripts/extract_humgut_phoraged.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=extract_phoraged
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

mkdir -p "$PROJECT/logs/slurm"

"$PYTHON_BIN" "$PROJECT/scripts/extract_humgut_phoraged.py" --shard-index "${SLURM_ARRAY_TASK_ID:-0}"

echo "Job finished at $(date)"
