#!/usr/bin/env bash
# =============================================================================
# 01_kraken2_fmt_reads.sh
#
# Classifies each FMT sample's host-decontaminated paired reads against the
# HumGut kraken2 DB (config.HUMGUT_KRAKEN2_DB). The .report feeds Bracken
# (03_bracken_run.sh). One array task per sample in config.FMT_READS_SAMPLE_LIST.
# Reads are expected at <FMT_READS_DIR>/<sample>/r1.fastq.gz and r2.fastq.gz.
# Skips samples whose .report already exists.
#
# Usage:
#   sbatch --array=0-2 mock_datasets/scripts/01_kraken2_fmt_reads.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=kraken2_fmt_reads
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
KRAKEN2_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/kraken2env/bin/kraken2"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

cfg() { "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.$1)"; }
KRAKEN2_DB=$(cfg HUMGUT_KRAKEN2_DB)
READS_DIR=$(cfg FMT_READS_DIR)
SAMPLE_LIST=$(cfg FMT_READS_SAMPLE_LIST)
OUT_DIR=$(cfg OUT_KRAKEN2_FMT_READS)

mapfile -t SAMPLES < "$SAMPLE_LIST"
SAMPLE="${SAMPLES[${SLURM_ARRAY_TASK_ID:-0}]}"

SAMPLE_OUT="$OUT_DIR/$SAMPLE"
mkdir -p "$SAMPLE_OUT"

if [ -f "$SAMPLE_OUT/${SAMPLE}.report" ]; then
    echo "$SAMPLE already classified, skipping."
    exit 0
fi

"$KRAKEN2_BIN" \
    --db "$KRAKEN2_DB" \
    --threads "${SLURM_CPUS_PER_TASK:-16}" \
    --paired \
    --use-names \
    --output "$SAMPLE_OUT/${SAMPLE}.kraken" \
    --report "$SAMPLE_OUT/${SAMPLE}.report" \
    "$READS_DIR/$SAMPLE/r1.fastq.gz" "$READS_DIR/$SAMPLE/r2.fastq.gz"
