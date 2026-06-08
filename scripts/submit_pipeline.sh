#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh
#
# Submits all SLURM jobs needed to build the spike-in benchmark datasets.
# Jobs are submitted with dependencies so they run in the correct order:
#
#   Step 1 — Read mixing (all 6 sample types run in parallel)
#     Simulates Illumina reads from viral references, mixes with background,
#     and outputs gzipped paired-end FASTQ files.
#
#   Step 2 — Assembly (waits for all Step 1 jobs to finish)
#     Assembles every FASTQ pair with MegaHit to produce assembly.fa files
#     that can be used directly by viral detection tools.
#
# Expected outputs:
#   data/spike_in_samples/{type}/sample{N}/{coverage}/sample.1.fq.gz
#   data/spike_in_samples/{type}/sample{N}/{coverage}/sample.2.fq.gz
#   data/spike_in_samples/{type}/sample{N}/{coverage}/assembly.fa
#
# Prerequisites:
#   1. pick_refs.py has been run (creates contigs.fasta in each sample dir)
#   2. MegaHit is installed: conda install -n test_env -c bioconda megahit=1.2.9 -y
#
# Usage:
#   bash scripts/submit_pipeline.sh
# =============================================================================

set -euo pipefail

SCRIPTS="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/scripts"
SPIKE_IN="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/spike_in_samples"
LOG_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm"

mkdir -p "$LOG_DIR"

# =============================================================================
# Step 1: Read mixing — one array job per sample type, 15 samples each
# =============================================================================
echo "=== Step 1: Submitting read mixing jobs ==="

jid_single=$(sbatch --array=0-14 --export=ALL,SAMPLES_DIR="${SPIKE_IN}/single"   --parsable "$SCRIPTS/read_mixing.sh")
jid_eq2=$(   sbatch --array=0-14 --export=ALL,SAMPLES_DIR="${SPIKE_IN}/equal2"   --parsable "$SCRIPTS/read_mixing.sh")
jid_eq3=$(   sbatch --array=0-14 --export=ALL,SAMPLES_DIR="${SPIKE_IN}/equal3"   --parsable "$SCRIPTS/read_mixing.sh")
jid_eq4=$(   sbatch --array=0-14 --export=ALL,SAMPLES_DIR="${SPIKE_IN}/equal4"   --parsable "$SCRIPTS/read_mixing.sh")
jid_uneq2=$( sbatch --array=0-14 --export=ALL,SAMPLES_DIR="${SPIKE_IN}/unequal2" --parsable "$SCRIPTS/read_mixing.sh")
jid_uneq3=$( sbatch --array=0-14 --export=ALL,SAMPLES_DIR="${SPIKE_IN}/unequal3" --parsable "$SCRIPTS/read_mixing.sh")

echo "  single:   job $jid_single"
echo "  equal2:   job $jid_eq2"
echo "  equal3:   job $jid_eq3"
echo "  equal4:   job $jid_eq4"
echo "  unequal2: job $jid_uneq2"
echo "  unequal3: job $jid_uneq3"

all_mixing="${jid_single}:${jid_eq2}:${jid_eq3}:${jid_eq4}:${jid_uneq2}:${jid_uneq3}"

# =============================================================================
# Step 2: Assembly — waits for all read mixing jobs to finish
# =============================================================================
echo ""
echo "=== Step 2: Submitting assembly jobs (depends on Step 1) ==="

# First build a work list of all FASTQ files to assemble.
# Total expected: 15*4 single + 15*4 equal2 + 15*4 equal3 + 15*4 equal4
#                 + 15*2 unequal2 + 15*2 unequal3 = 300 FASTQ pairs
WORK_LIST="${SPIKE_IN}/assembly_work_list.txt"

jid_setup=$(sbatch \
    --partition=compute \
    --job-name=build_work_list \
    --output="${LOG_DIR}/setup_%j.out" \
    --time=00:05:00 --mem=1G --ntasks=1 \
    --dependency="afterok:${all_mixing}" \
    --parsable \
    --wrap "find ${SPIKE_IN} -name 'sample.1.fq.gz' | sort > ${WORK_LIST} && echo \"Work list: \$(wc -l < ${WORK_LIST}) FASTQ pairs to assemble\"")

jid_assemble=$(sbatch \
    --array=0-299 \
    --dependency="afterok:${jid_setup}" \
    --parsable \
    "$SCRIPTS/assemble.sh")

echo "  work list setup: job $jid_setup"
echo "  assembly:        job $jid_assemble"

echo ""
echo "All jobs submitted. Monitor progress with: squeue -u \$USER"
