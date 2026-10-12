#!/usr/bin/env bash
# =============================================================================
# kraken2_fmt.sh
#
# Classifies real FMT sample assemblies (SPAdes contigs.fasta, from the
# separate fmt_viruses repo) against the custom HumGut kraken2 database
# (config.HUMGUT_KRAKEN2_DB, built 2026-09-08, no human library) to give each
# contig a HumGut taxon assignment. This is the first step toward building a
# real FMT abundance profile for InSilicoSeq (per-genome sequencing depth =
# sum(contig length x SPAdes coverage) over each genome's assigned contigs,
# normalized -- computed downstream by build_fmt_abundance_profile.py, not
# here): kraken2 needs to run first so each contig has a HumGut genome
# assignment to sum coverage against.
#
# Classifies contigs (not raw reads) directly, one array task per sample in
# config.FMT_SAMPLE_LIST (currently data/sample_lists/xac: ERR2198611-613).
# Kraken2's --output keeps each record's own FASTA header as its read ID, so
# SPAdes contig IDs (NODE_<n>_length_<len>_cov_<cov>) survive untouched for
# the downstream coverage join.
#
# Idempotent -- skips any sample whose .report already exists.
#
# Usage:
#   sbatch --array=0-2 scripts/kraken2_fmt.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=kraken2_fmt
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
ASSEMBLIES_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_ASSEMBLIES_DIR)")
SAMPLE_LIST=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_SAMPLE_LIST)")
OUT_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.OUT_KRAKEN2_FMT)")

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

CONTIGS="$ASSEMBLIES_DIR/$SAMPLE/contigs.fasta"
if [ ! -f "$CONTIGS" ]; then
    echo "ERROR: contigs.fasta not found for $SAMPLE at $CONTIGS"
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

echo "Classifying $CONTIGS against $KRAKEN2_DB"
"$KRAKEN2_BIN" \
    --db "$KRAKEN2_DB" \
    --threads "${SLURM_CPUS_PER_TASK:-16}" \
    --use-names \
    --output "$KRAKEN_OUTPUT" \
    --report "$KRAKEN_REPORT" \
    "$CONTIGS"

echo "Done. Output: $KRAKEN_OUTPUT"
echo "Report: $KRAKEN_REPORT"
echo "Job finished at $(date)"
