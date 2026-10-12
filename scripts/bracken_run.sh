#!/usr/bin/env bash
# =============================================================================
# bracken_run.sh
#
# Runs Bracken against each real FMT sample's kraken2 --report (from
# scripts/kraken2_fmt_reads.sh) to redistribute reads that landed at an
# ancestor rank (genus/family/order/...) back down onto the specific HumGut
# genomes actually detected -- see NOTEBOOK.md 2026-09-19 for the measured
# problem (32.2% of ERR2198703's classified reads were stuck above genome
# level) this corrects for. Uses the kmer-distribution index built by
# scripts/bracken_build.sh (database100mers.kmer_distrib, at read length 100,
# level S -- this DB's taxonomy assigns each HumGut genome its own "S"-rank
# node, confirmed 2026-09-19, so level S here means per-genome, not
# per-species).
#
# One array task per sample in config.FMT_READS_SAMPLE_LIST. Output is
# written alongside each sample's existing kraken2 output:
#   dataset_creation/kraken2_fmt_reads/<sample>/<sample>.bracken
#     -- per-genome corrected read counts + abundance fractions
#   dataset_creation/kraken2_fmt_reads/<sample>/<sample>_bracken.report
#     -- kraken2-report-style file with Bracken's corrected counts, for
#        anything downstream that expects the standard report format
#
# Idempotent -- skips any sample whose .bracken output already exists.
#
# Usage:
#   sbatch --array=0-2 scripts/bracken_run.sh
# =============================================================================
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

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

set -euo pipefail

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_bracken:3.1--h9948957_0.sif"

module load apptainer/1.4.2-1

# Loaded from config.py (single source of truth) rather than hardcoded here.
KRAKEN2_DB=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.HUMGUT_KRAKEN2_DB)")
SAMPLE_LIST=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_READS_SAMPLE_LIST)")
OUT_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.OUT_KRAKEN2_FMT_READS)")

if [ ! -f "$KRAKEN2_DB/database100mers.kmer_distrib" ]; then
    echo "ERROR: Bracken index not found at $KRAKEN2_DB/database100mers.kmer_distrib -- run scripts/bracken_build.sh first."
    exit 1
fi
if [ ! -f "$SAMPLE_LIST" ]; then
    echo "ERROR: sample list not found at $SAMPLE_LIST"
    exit 1
fi

mapfile -t SAMPLES < "$SAMPLE_LIST"
if [ "${#SAMPLES[@]}" -eq 0 ]; then
    echo "ERROR: $SAMPLE_LIST is empty."
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#SAMPLES[@]}" ]; then
    echo "ERROR: array task $TASK_ID out of range (${#SAMPLES[@]} samples). Resubmit with --array=0-$((${#SAMPLES[@]} - 1))."
    exit 1
fi
SAMPLE="${SAMPLES[$TASK_ID]}"
echo "Sample: $SAMPLE"

SAMPLE_OUT="$OUT_DIR/$SAMPLE"
KRAKEN_REPORT="$SAMPLE_OUT/${SAMPLE}.report"
if [ ! -f "$KRAKEN_REPORT" ]; then
    echo "ERROR: kraken2 report not found for $SAMPLE at $KRAKEN_REPORT -- run scripts/kraken2_fmt_reads.sh first."
    exit 1
fi

BRACKEN_OUT="$SAMPLE_OUT/${SAMPLE}.bracken"
BRACKEN_REPORT="$SAMPLE_OUT/${SAMPLE}_bracken.report"

if [ -f "$BRACKEN_OUT" ]; then
    echo "Bracken output already exists for $SAMPLE at $BRACKEN_OUT -- skipping."
    exit 0
fi

echo "Running bracken on $KRAKEN_REPORT (level S, read length 100)"
apptainer exec --bind /rs1,/gpfs_common "$SIF" \
    bracken -d "$KRAKEN2_DB" \
    -i "$KRAKEN_REPORT" \
    -o "$BRACKEN_OUT" \
    -w "$BRACKEN_REPORT" \
    -r 100 \
    -l S \
    -t 10

echo "Done. Output: $BRACKEN_OUT"
echo "Corrected report: $BRACKEN_REPORT"
echo "Job finished at $(date)"
