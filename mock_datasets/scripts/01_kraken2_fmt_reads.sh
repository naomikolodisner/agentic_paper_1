#!/usr/bin/env bash
# =============================================================================
# 01_kraken2_fmt_reads.sh
#
# Classifies real FMT sample RAW READS (not the SPAdes assembly -- see
# mock_datasets/scripts/kraken2_fmt.sh for the superseded contig-based pass) against the
# custom HumGut kraken2 database (config.HUMGUT_KRAKEN2_DB, built 2026-09-08,
# no human library) to build a proper read-count-based abundance profile.
#
# Switched from contigs to reads because ERR2198611-613 (kraken2_fmt.sh's
# sample set) never had their raw reads land on Hazel -- only their SPAdes
# assembly did (source reads lived on a now-inaccessible external /xdisk
# cluster). ERR2198703-705 (config.FMT_READS_SAMPLE_LIST -> xae) DO have
# their actual host(human)-decontaminated paired reads on Hazel already
# (bowtie2 --un-conc-gz vs. CHM13, by the upstream fmt_viruses repo --
# fmt_viruses/03_contam_removal/run_remove_human.sh) -- no assembly-coverage
# proxy needed for these; kraken2's own per-read classification against a
# genome-level (not just species-level) HumGut taxonomy is a direct depth
# signal once summed per genome.
#
# One array task per sample in config.FMT_READS_SAMPLE_LIST. Uses kraken2
# --paired against a sample's <sample>_1.fastq.gz/_2.fastq.gz pair.
#
# NOTE: this produces the classification + kraken2 --report only. Standard
# practice is to follow up with Bracken to redistribute reads whose LCA
# collapsed to a rank above individual HumGut genome back down to genome
# level -- not yet decided/built (requires a one-time, compute-heavy
# bracken-build pass against this DB at this read length). Until that
# lands, treat the raw --report per-genome read counts as an approximation.
#
# Idempotent -- skips any sample whose .report already exists.
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

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

set -euo pipefail

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
KRAKEN2_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/kraken2env/bin/kraken2"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

# Loaded from config.py (single source of truth) rather than hardcoded here.
KRAKEN2_DB=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.HUMGUT_KRAKEN2_DB)")
READS_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_READS_DIR)")
SAMPLE_LIST=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_READS_SAMPLE_LIST)")
OUT_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.OUT_KRAKEN2_FMT_READS)")

mkdir -p "$PROJECT/logs/slurm" "$OUT_DIR"

# --- Validate prerequisites ---------------------------------------------------

if [ ! -x "$KRAKEN2_BIN" ]; then
    echo "ERROR: kraken2 binary not found/executable at $KRAKEN2_BIN"
    exit 1
fi
if [ ! -f "$KRAKEN2_DB/hash.k2d" ]; then
    echo "ERROR: kraken2 DB not found/incomplete at $KRAKEN2_DB (missing hash.k2d)"
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

READ1="$READS_DIR/${SAMPLE}_1.fastq.gz"
READ2="$READS_DIR/${SAMPLE}_2.fastq.gz"
if [ ! -f "$READ1" ] || [ ! -f "$READ2" ]; then
    echo "ERROR: paired reads not found for $SAMPLE ($READ1 / $READ2)"
    exit 1
fi

SAMPLE_OUT="$OUT_DIR/$SAMPLE"
mkdir -p "$SAMPLE_OUT"
KRAKEN_OUTPUT="$SAMPLE_OUT/${SAMPLE}.kraken"
KRAKEN_REPORT="$SAMPLE_OUT/${SAMPLE}.report"

if [ -f "$KRAKEN_REPORT" ]; then
    echo "Report already exists for $SAMPLE at $KRAKEN_REPORT -- skipping."
    exit 0
fi

echo "Classifying $READ1 / $READ2 against $KRAKEN2_DB"
"$KRAKEN2_BIN" \
    --db "$KRAKEN2_DB" \
    --threads "${SLURM_CPUS_PER_TASK:-16}" \
    --paired \
    --use-names \
    --output "$KRAKEN_OUTPUT" \
    --report "$KRAKEN_REPORT" \
    "$READ1" "$READ2"

echo "Done. Output: $KRAKEN_OUTPUT"
echo "Report: $KRAKEN_REPORT"
echo "Job finished at $(date)"
