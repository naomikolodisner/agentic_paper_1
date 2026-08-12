#!/usr/bin/env bash
# =============================================================================
# assemble.sh
#
# Assembles spike-in FASTQ files into contigs using MegaHit, producing an
# assembly.fa alongside each percentage tier's final FASTQ. This is the
# final step in spike-in sample creation and makes the data usable by viral
# detection tools.
#
# This script is meant to be run as a SLURM array job over a work list of
# FASTQ files. The work list is generated automatically by submit_pipeline.sh
# after all read mixing jobs complete.
#
# Assembly settings match the set3_simulated_metagenomes benchmark dataset:
#   - MegaHit v1.2.9
#   - Minimum contig length: 1000 bp
#
# Output: assembly.fa alongside each final.1.fq.gz / final.2.fq.gz pair
#   e.g. data/spike_in_samples/sample_10v_3/spike_1pct/assembly.fa
#
# The script is idempotent — it skips any sample that already has assembly.fa.
#
# Prerequisites:
#   MegaHit must be installed:
#     conda install -n test_env -c bioconda megahit=1.2.9 -y
#
#   read_mixing.sh must have completed for all sample dirs.
#
# This script is normally submitted by submit_pipeline.sh, not directly.
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=assemble
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
MEGAHIT_BIN="$PROJECT/../conda_envs/test_env/bin/megahit"
WORK_LIST="${OVERRIDE_WORK_LIST:-$PROJECT/data/spike_in_samples/assembly_work_list.txt}"

# --- Validate prerequisites --------------------------------------------------

if [ ! -f "$MEGAHIT_BIN" ]; then
    echo "ERROR: megahit not found at $MEGAHIT_BIN"
    echo "Install with: conda install -n test_env -c bioconda megahit=1.2.9 -y"
    exit 1
fi

if [ ! -f "$WORK_LIST" ]; then
    echo "ERROR: Work list not found: $WORK_LIST"
    echo "This file is created automatically by submit_pipeline.sh after read mixing completes."
    exit 1
fi

# --- Select this array task's FASTQ pair -------------------------------------

mapfile -t R1_FILES < "$WORK_LIST"

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#R1_FILES[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds work list size (${#R1_FILES[@]}). Nothing to do."
    exit 0
fi

r1="${R1_FILES[$SLURM_ARRAY_TASK_ID]}"
r2="${r1/final.1.fq.gz/final.2.fq.gz}"
out_dir=$(dirname "$r1")
assembly_out="${out_dir}/assembly.fa"

# --- Skip if already assembled -----------------------------------------------

if [ -f "$assembly_out" ]; then
    echo "Skipping — assembly already exists: $assembly_out"
    exit 0
fi

if [ ! -f "$r1" ] || [ ! -f "$r2" ]; then
    echo "ERROR: FASTQ files not found: $r1"
    exit 1
fi

echo "Assembling: $out_dir"

# --- Run MegaHit -------------------------------------------------------------

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

"$MEGAHIT_BIN" \
    -1 "$r1" \
    -2 "$r2" \
    --min-contig-len 1000 \
    --num-cpu-threads 8 \
    --memory 100000000000 \
    -o "$TMPDIR/megahit_out"

if [ ! -f "$TMPDIR/megahit_out/final.contigs.fa" ]; then
    echo "ERROR: MegaHit failed for $out_dir"
    exit 1
fi

cp "$TMPDIR/megahit_out/final.contigs.fa" "$assembly_out"
echo "Assembly written: $assembly_out"
echo "Finished at $(date)"
