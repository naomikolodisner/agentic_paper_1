#!/usr/bin/env bash
# Submits all read mixing and assembly jobs for spike-in samples.
# Run from the project root: bash scripts/submit_spike_in_pipeline.sh
#
# Step 1: finish read mixing (missing single + all equal + all unequal)
# Step 2: assemble everything with MegaHit (depends on Step 1)
#
# Prerequisites:
#   conda install -n test -c bioconda megahit=1.2.9 -y

set -euo pipefail

SCRIPTS="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/scripts"
SPIKE_IN="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/spike_in_samples"
LOG_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm"

mkdir -p "$LOG_DIR"

echo "=== Step 1: Read mixing ==="

# single samples (skip already-done ones via idempotency check in the script)
jid_single=$(sbatch --array=0-14 \
    --export=SAMPLES_DIR="${SPIKE_IN}/single" \
    --parsable \
    "$SCRIPTS/finish_read_mixing.sh")
echo "single:   job $jid_single"

jid_eq2=$(sbatch --array=0-14 \
    --export=SAMPLES_DIR="${SPIKE_IN}/equal2" \
    --parsable \
    "$SCRIPTS/finish_read_mixing.sh")
echo "equal2:   job $jid_eq2"

jid_eq3=$(sbatch --array=0-14 \
    --export=SAMPLES_DIR="${SPIKE_IN}/equal3" \
    --parsable \
    "$SCRIPTS/finish_read_mixing.sh")
echo "equal3:   job $jid_eq3"

jid_eq4=$(sbatch --array=0-14 \
    --export=SAMPLES_DIR="${SPIKE_IN}/equal4" \
    --parsable \
    "$SCRIPTS/finish_read_mixing.sh")
echo "equal4:   job $jid_eq4"

jid_uneq2=$(sbatch --array=0-14 \
    --export=SAMPLES_DIR="${SPIKE_IN}/unequal2",UNEQUAL_TYPE=2 \
    --parsable \
    "$SCRIPTS/create_unequal_samples_slurm.sh")
echo "unequal2: job $jid_uneq2"

jid_uneq3=$(sbatch --array=0-14 \
    --export=SAMPLES_DIR="${SPIKE_IN}/unequal3",UNEQUAL_TYPE=3 \
    --parsable \
    "$SCRIPTS/create_unequal_samples_slurm.sh")
echo "unequal3: job $jid_uneq3"

# Collect all mixing job IDs for dependency
all_mixing="${jid_single}:${jid_eq2}:${jid_eq3}:${jid_eq4}:${jid_uneq2}:${jid_uneq3}"

echo ""
echo "=== Step 2: Assembly (depends on all mixing jobs) ==="

# Build the work list after mixing completes using an inline --wrap command
WORK_LIST="${SPIKE_IN}/assembly_work_list.txt"

jid_setup=$(sbatch \
    --partition=compute \
    --job-name=build_work_list \
    --output="${LOG_DIR}/setup_%j.out" \
    --time=00:05:00 \
    --mem=1G \
    --ntasks=1 \
    --dependency="afterok:${all_mixing}" \
    --parsable \
    --wrap "find ${SPIKE_IN} -name 'sample.1.fq.gz' | sort > ${WORK_LIST} && echo \"Work list: \$(wc -l < ${WORK_LIST}) assemblies to run\"")
echo "work list setup: job $jid_setup"

# Submit assembly array — size will be determined at submission time after setup
# We know the maximum: 11 existing + 4 new single + 60 equal2 + 60 equal3 + 60 equal4 + 30 unequal2 + 30 unequal3 = 255
# Use 0-254 and let the script skip out-of-range indices
jid_assemble=$(sbatch --array=0-254 \
    --dependency="afterok:${jid_setup}" \
    --parsable \
    "$SCRIPTS/assemble_spike_ins.sh")
echo "assembly: job $jid_assemble"

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
echo "Assembly outputs will be: data/spike_in_samples/{type}/sample{N}/{cov}/assembly.fa"
