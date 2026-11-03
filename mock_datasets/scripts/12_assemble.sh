#!/usr/bin/env bash
# Assemble each spike-in read pair with MegaHit (min contig length 1000) into assembly.fa next to the reads.
# Usage: find data/spike_in_samples -name final.1.fq.gz | sort > data/spike_in_samples/assembly_work_list.txt
#        sbatch --array=0-$(($(wc -l < data/spike_in_samples/assembly_work_list.txt) - 1)) mock_datasets/scripts/12_assemble.sh
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

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
MEGAHIT_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/megahit"
WORK_LIST="${OVERRIDE_WORK_LIST:-$PROJECT/data/spike_in_samples/assembly_work_list.txt}"

mapfile -t R1_FILES < "$WORK_LIST"
r1="${R1_FILES[$SLURM_ARRAY_TASK_ID]}"
out=$(dirname "$r1")

[ -f "$out/assembly.fa" ] && { echo "$out done, skipping"; exit 0; }

TMP="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

"$MEGAHIT_BIN" \
    -1 "$r1" \
    -2 "${r1/final.1.fq.gz/final.2.fq.gz}" \
    --min-contig-len 1000 \
    --num-cpu-threads "${SLURM_CPUS_PER_TASK:-8}" \
    --memory 100000000000 \
    -o "$TMP/megahit_out"

cp "$TMP/megahit_out/final.contigs.fa" "$out/assembly.fa"
