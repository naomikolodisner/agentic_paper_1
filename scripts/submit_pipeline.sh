#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh
#
# Submits all SLURM jobs needed to build the spike-in benchmark datasets.
# Jobs are submitted with dependencies so they run in the correct order:
#
#   Step 0 — Background generation (4-combo array: config.ISS_MODELS [hiseq,
#     novaseq] x 2 abundance distributions x 1 read depth). Simulates the
#     virus-free background metagenome pool with InSilicoSeq from a random
#     subset of raw HumGut MAGs (prophages intact -- PHORAGER hasn't been run
#     on HumGut yet; extract_humgut_subset.py extracts straight from
#     HumGut2.tar as a stand-in until PHORAGER output lands).
#
#   Step 1 — Read mixing (one array job, 3 sample dirs; depends on Step 0)
#     For each sample dir's fixed viral subset, builds EVERY combination of
#     percentage (config.SPIKE_PERCENTAGES: 1%/5%/10%) x ART-compatible
#     background row (exhaustive, not a random per-combo pick) by simulating
#     viral spike-in reads with ART Illumina (coverage split unevenly across
#     the subset via ART_RATIO_WEIGHTS) and mixing them with that background.
#
#   Step 1.5 — Kraken2 check (depends on Step 1)
#     Confirms every viral reference actually spiked into the generated
#     samples is detectable by kraken2, using the real post-mix reads
#     (pooled by source genome ID, which combine_reads.py's shuffle leaves
#     untouched) -- before the heavier assembly step below runs on top of them.
#
#   Step 2 — Assembly (depends on Step 1.5)
#     Assembles every FASTQ pair with MegaHit to produce assembly.fa files
#     that can be used directly by viral detection tools.
#
# Expected outputs:
#   data/spike_in_samples/sample_{size}v_{n}/spike_{pct}pct_{bg_combo}/final.1.fq.gz
#   data/spike_in_samples/sample_{size}v_{n}/spike_{pct}pct_{bg_combo}/final.2.fq.gz
#   data/spike_in_samples/sample_{size}v_{n}/spike_{pct}pct_{bg_combo}/assembly.fa
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
# Step 0: Background generation — 4-combo array
# =============================================================================
echo "=== Step 0: Submitting background generation jobs ==="

jid_background=$(sbatch --array=0-3 --parsable "$SCRIPTS/generate_background.sh")
echo "  background: job $jid_background"

# =============================================================================
# Step 1: Read mixing — one array job, 15 sample dirs, depends on Step 0
# =============================================================================
echo ""
echo "=== Step 1: Submitting read mixing job (depends on Step 0) ==="

jid_mixing=$(sbatch \
    --array=0-2 \
    --dependency="afterok:${jid_background}" \
    --parsable \
    "$SCRIPTS/read_mixing.sh")
echo "  read mixing: job $jid_mixing"

# =============================================================================
# Step 1.5: Kraken2 check — depends on Step 1
# =============================================================================
echo ""
echo "=== Step 1.5: Submitting kraken2 check (depends on Step 1) ==="

jid_kraken2=$(sbatch \
    --dependency="afterok:${jid_mixing}" \
    --parsable \
    "$SCRIPTS/kraken2_check.sh")
echo "  kraken2 check: job $jid_kraken2"

# =============================================================================
# Step 2: Assembly — waits for kraken2 check to pass
# =============================================================================
echo ""
echo "=== Step 2: Submitting assembly jobs (depends on Step 1.5) ==="

# Build a work list of all FASTQ files to assemble.
# Expected: 3 sample dirs x 3 percentages x 4 background rows = 36 FASTQ pairs
WORK_LIST="${SPIKE_IN}/assembly_work_list.txt"

jid_setup=$(sbatch \
    --partition=compute \
    --job-name=build_work_list \
    --output="${LOG_DIR}/setup_%j.out" \
    --time=00:05:00 --mem=1G --ntasks=1 \
    --dependency="afterok:${jid_kraken2}" \
    --mail-type=FAIL --mail-user=nkolodi@ncsu.edu \
    --parsable \
    --wrap "find ${SPIKE_IN} -name 'final.1.fq.gz' | sort > ${WORK_LIST} && echo \"Work list: \$(wc -l < ${WORK_LIST}) FASTQ pairs to assemble\"")

jid_assemble=$(sbatch \
    --array=0-35 \
    --dependency="afterok:${jid_setup}" \
    --parsable \
    "$SCRIPTS/assemble.sh")

echo "  work list setup: job $jid_setup"
echo "  assembly:        job $jid_assemble"

echo ""
echo "All jobs submitted. Monitor progress with: squeue -u \$USER"
