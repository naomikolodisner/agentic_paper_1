#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --job-name=read_mixing
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# Submit with: sbatch --array=0-14 --export=SAMPLES_DIR=/path/to/type finish_read_mixing.sh

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

if [ -z "$SAMPLES_DIR" ]; then
    echo "ERROR: SAMPLES_DIR not set. Use --export=SAMPLES_DIR=..."
    exit 1
fi

GEN_TITRATION="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/scripts/gen_titration_sample.py"
BACKGROUND_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/no_virus_contigs"
ART_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/art_illumina"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

mkdir -p /rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm

mapfile -t SAMPLE_DIRS < <(find "$SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SAMPLE_DIRS[@]}" ]; then
    echo "Task ID ${SLURM_ARRAY_TASK_ID} exceeds number of samples (${#SAMPLE_DIRS[@]}). Exiting."
    exit 0
fi

sample_dir="${SAMPLE_DIRS[$SLURM_ARRAY_TASK_ID]}"
sample_name=$(basename "$sample_dir")

echo "Processing: $sample_name"

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

export OMP_NUM_THREADS=1

contigs_fasta="${sample_dir}/contigs.fasta"
if [ ! -f "$contigs_fasta" ]; then
    echo "ERROR: Missing $contigs_fasta"
    exit 1
fi

coverages=(0.1 0.5 1 10)

for cov in "${coverages[@]}"; do
    out_dir="${sample_dir}/${cov}x"

    if [ -f "${out_dir}/sample.1.fq.gz" ] && [ -f "${out_dir}/sample.2.fq.gz" ]; then
        echo "Skipping ${sample_name} ${cov}x — already complete"
        continue
    fi

    echo "Coverage ${cov}x"
    mkdir -p "$out_dir"

    "$ART_BIN" -ss HS25 -na \
        -i "$contigs_fasta" \
        -f "$cov" \
        -l 100 -m 200 -s 10 \
        -o "$TMPDIR/sample"

    if [ ! -f "$TMPDIR/sample1.fq" ]; then
        echo "ERROR: ART failed for ${sample_name} ${cov}x"
        exit 1
    fi

    mv "$TMPDIR/sample1.fq" "$TMPDIR/sample.1.fq"
    mv "$TMPDIR/sample2.fq" "$TMPDIR/sample.2.fq"

    background_fasta=$(ls "${BACKGROUND_DIR}"/*/contigs.fasta | shuf -n 1)

    "$ART_BIN" \
        -i "$background_fasta" \
        -f 10 \
        -l 100 -m 200 -s 10 \
        -o "$TMPDIR/background"

    if [ ! -f "$TMPDIR/background1.fq" ]; then
        echo "ERROR: ART background failed for ${sample_name} ${cov}x"
        rm -f "$TMPDIR/sample.1.fq" "$TMPDIR/sample.2.fq"
        continue
    fi

    "$PYTHON_BIN" "$GEN_TITRATION" \
        -R1 "$TMPDIR/sample.1.fq" \
        -R2 "$TMPDIR/sample.2.fq" \
        -B1 "$TMPDIR/background1.fq" \
        -B2 "$TMPDIR/background2.fq" \
        --depth 3000000000 \
        -o "${out_dir}/sample"

    rm -f "$TMPDIR"/*.fq
    echo "Done: ${sample_name} ${cov}x"
done

echo "Finished ${sample_name} at $(date)"
