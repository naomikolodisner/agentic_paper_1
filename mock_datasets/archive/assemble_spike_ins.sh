#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --job-name=assemble_spike_ins
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Array size is set by submit_spike_in_pipeline.sh based on the work list

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

SPIKE_IN_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/spike_in_samples"
MEGAHIT_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/megahit"
WORK_LIST="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/spike_in_samples/assembly_work_list.txt"

if [ ! -f "$MEGAHIT_BIN" ]; then
    echo "ERROR: megahit not found at $MEGAHIT_BIN"
    echo "Install with: conda install -n test -c bioconda megahit=1.2.9 -y"
    exit 1
fi

if [ ! -f "$WORK_LIST" ]; then
    echo "ERROR: Work list not found: $WORK_LIST"
    echo "Run submit_spike_in_pipeline.sh to generate it"
    exit 1
fi

mapfile -t R1_FILES < "$WORK_LIST"

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#R1_FILES[@]}" ]; then
    echo "Task ID exceeds work list size. Exiting."
    exit 0
fi

r1="${R1_FILES[$SLURM_ARRAY_TASK_ID]}"
r2="${r1/sample.1.fq.gz/sample.2.fq.gz}"
out_dir=$(dirname "$r1")
assembly_out="${out_dir}/assembly.fa"

if [ -f "$assembly_out" ]; then
    echo "Skipping — assembly already exists: $assembly_out"
    exit 0
fi

if [ ! -f "$r1" ] || [ ! -f "$r2" ]; then
    echo "ERROR: FASTQ files missing: $r1"
    exit 1
fi

echo "Assembling: $out_dir"

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

"$MEGAHIT_BIN" \
    -1 "$r1" \
    -2 "$r2" \
    --min-contig-len 1000 \
    --num-cpu-threads 8 \
    -o "$TMPDIR/megahit_out"

if [ ! -f "$TMPDIR/megahit_out/final.contigs.fa" ]; then
    echo "ERROR: MegaHit failed for $out_dir"
    exit 1
fi

cp "$TMPDIR/megahit_out/final.contigs.fa" "$assembly_out"
echo "Assembly written: $assembly_out"
echo "Finished at $(date)"
