#!/usr/bin/env bash
# Extract the needed phoraged HumGut genomes, one array task per tarball shard (7 shards).
# Usage: sbatch --array=0-6 mock_datasets/scripts/05_extract_humgut_phoraged.sh
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

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

"$PYTHON_BIN" "$PROJECT/mock_datasets/scripts/helpers/extract_humgut_phoraged.py" --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
