#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh
#
# Submits all SLURM jobs needed to build the ISS-based spike-in benchmark
# datasets. Jobs are submitted with dependencies so they run in the correct
# order:
#
#   Step 0 — Background generation (48-combo array: 4 models x 4 abundance
#     distributions x 3 read depths). Simulates the virus-free background
#     metagenome pool from PHORAGER's prophage-stripped HumGut MAGs. If that
#     HumGut/PHORAGER output doesn't exist yet, every task in this step fails
#     fast with a clear message -- the whole chain below then stays pending
#     (SLURM cancels afterok-dependent jobs when their dependency fails), so
#     this is a safe no-op to submit before that data lands. Re-run once it
#     does.
#
#   Step 0.5 — Kraken2 sanity check (depends on Step 0)
#     Confirms every viral reference used across the generated spike-in
#     samples is actually detectable by kraken2, before the heavier steps
#     below run on top of them.
#
#   Step 1 — Read mixing (one array job, 15 sample dirs; depends on Step 0.5)
#     For each sample dir's fixed viral subset, builds every
#     distribution x percentage combo (16 per sample) by simulating viral
#     spike-in reads with ISS and mixing them with a random background.
#
#   Step 2 — Assembly (waits for Step 1)
#     Assembles every FASTQ pair with MegaHit to produce assembly.fa files
#     that can be used directly by viral detection tools.
#
# Expected outputs:
#   data/spike_in_samples/sample_{size}v_{n}/{dist}/spike_{pct}pct/final.1.fq.gz
#   data/spike_in_samples/sample_{size}v_{n}/{dist}/spike_{pct}pct/final.2.fq.gz
#   data/spike_in_samples/sample_{size}v_{n}/{dist}/spike_{pct}pct/assembly.fa
#
# Prerequisites:
#   1. pick_refs.py has been run (creates contigs.fasta in each sample dir)
#   2. PHORAGER has been run on HumGut (see config.py: HUMGUT_PROPHAGE_REMOVED_DIR)
#   3. MegaHit is installed: conda install -n test_env -c bioconda megahit=1.2.9 -y
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
# Step 0: Background generation — 48-combo array
# =============================================================================
echo "=== Step 0: Submitting background generation jobs ==="

jid_background=$(sbatch --array=0-47 --parsable "$SCRIPTS/generate_background.sh")
echo "  background: job $jid_background"

# =============================================================================
# Step 0.5: Kraken2 sanity check — depends on Step 0
# =============================================================================
echo ""
echo "=== Step 0.5: Submitting kraken2 sanity check (depends on Step 0) ==="

jid_kraken2=$(sbatch \
    --dependency="afterok:${jid_background}" \
    --parsable \
    "$SCRIPTS/kraken2_sanity_check.sh")
echo "  kraken2 sanity check: job $jid_kraken2"

# =============================================================================
# Step 1: Read mixing — one array job, 15 sample dirs, depends on Step 0.5
# =============================================================================
echo ""
echo "=== Step 1: Submitting read mixing job (depends on Step 0.5) ==="

jid_mixing=$(sbatch \
    --array=0-14 \
    --dependency="afterok:${jid_kraken2}" \
    --parsable \
    "$SCRIPTS/read_mixing.sh")
echo "  read mixing: job $jid_mixing"

# =============================================================================
# Step 2: Assembly — waits for read mixing to finish
# =============================================================================
echo ""
echo "=== Step 2: Submitting assembly jobs (depends on Step 1) ==="

# Build a work list of all FASTQ files to assemble.
# Expected: 15 sample dirs x 4 distributions x 4 percentages = 240 FASTQ pairs
WORK_LIST="${SPIKE_IN}/assembly_work_list.txt"

jid_setup=$(sbatch \
    --partition=compute \
    --job-name=build_work_list \
    --output="${LOG_DIR}/setup_%j.out" \
    --time=00:05:00 --mem=1G --ntasks=1 \
    --dependency="afterok:${jid_mixing}" \
    --parsable \
    --wrap "find ${SPIKE_IN} -name 'final.1.fq.gz' | sort > ${WORK_LIST} && echo \"Work list: \$(wc -l < ${WORK_LIST}) FASTQ pairs to assemble\"")

jid_assemble=$(sbatch \
    --array=0-239 \
    --dependency="afterok:${jid_setup}" \
    --parsable \
    "$SCRIPTS/assemble.sh")

echo "  work list setup: job $jid_setup"
echo "  assembly:        job $jid_assemble"

echo ""
echo "All jobs submitted. Monitor progress with: squeue -u \$USER"
