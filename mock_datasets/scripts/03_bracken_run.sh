#!/usr/bin/env bash
# Re-estimate each FMT sample's kraken2 report at level S with Bracken (per-genome in the HumGut DB).
# Usage: sbatch --array=0-2 mock_datasets/scripts/03_bracken_run.sh
#SBATCH --partition=compute
#SBATCH --job-name=bracken_run
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_bracken:3.1--h9948957_0.sif"
cfg() { "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.$1)"; }

DB=$(cfg HUMGUT_KRAKEN2_DB)
READ_LEN=100
mapfile -t SAMPLES < "$(cfg FMT_READS_SAMPLE_LIST)"
SAMPLE="${SAMPLES[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$(cfg OUT_KRAKEN2_FMT_READS)/$SAMPLE"
module load apptainer/1.4.2-1

[ -f "$OUT/$SAMPLE.bracken" ] && { echo "$SAMPLE done, skipping"; exit 0; }

apptainer exec --bind /rs1,/gpfs_common "$SIF" \
    bracken \
    -d "$DB" \
    -i "$OUT/$SAMPLE.report" \
    -o "$OUT/$SAMPLE.bracken" \
    -w "$OUT/${SAMPLE}_bracken.report" \
    -r "$READ_LEN" \
    -l S \
    -t 10
