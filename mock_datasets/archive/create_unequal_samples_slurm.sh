#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --job-name=unequal_mixing
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# Submit with:
#   sbatch --array=0-14 --export=SAMPLES_DIR=.../unequal2,UNEQUAL_TYPE=2 create_unequal_samples_slurm.sh
#   sbatch --array=0-14 --export=SAMPLES_DIR=.../unequal3,UNEQUAL_TYPE=3 create_unequal_samples_slurm.sh

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

if [ -z "$SAMPLES_DIR" ] || [ -z "$UNEQUAL_TYPE" ]; then
    echo "ERROR: SAMPLES_DIR and UNEQUAL_TYPE must be set via --export"
    exit 1
fi

GEN_TITRATION="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/scripts/gen_titration_sample.py"
BACKGROUND_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/no_virus_contigs"
ART_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/art_illumina"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

mkdir -p /rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm

mapfile -t SAMPLE_DIRS < <(find "$SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SAMPLE_DIRS[@]}" ]; then
    echo "Task ID exceeds number of samples. Exiting."
    exit 0
fi

sample_dir="${SAMPLE_DIRS[$SLURM_ARRAY_TASK_ID]}"
sample_name=$(basename "$sample_dir")

echo "Processing: $sample_name (unequal${UNEQUAL_TYPE})"

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

export OMP_NUM_THREADS=1

contigs_fasta="${sample_dir}/contigs.fasta"
if [ ! -f "$contigs_fasta" ]; then
    echo "ERROR: Missing $contigs_fasta"
    exit 1
fi

# Split contigs.fasta into one file per sequence
awk '/^>/{f=ENVIRON["TMPDIR"]"/ref"(++i)".fa"} {print > f}' "$contigs_fasta"

# Coverage ratio sets: each row is space-separated coverages for ref1 ref2 [ref3]
if [ "$UNEQUAL_TYPE" -eq 2 ]; then
    declare -a RATIO_NAMES=("10x_1x" "1x_0.5x")
    declare -a RATIO_SETS=("10 1" "1 0.5")
elif [ "$UNEQUAL_TYPE" -eq 3 ]; then
    declare -a RATIO_NAMES=("10x_1x_0.5x" "1x_0.5x_0.1x")
    declare -a RATIO_SETS=("10 1 0.5" "1 0.5 0.1")
else
    echo "ERROR: UNEQUAL_TYPE must be 2 or 3"
    exit 1
fi

for idx in "${!RATIO_NAMES[@]}"; do
    ratio_name="${RATIO_NAMES[$idx]}"
    out_dir="${sample_dir}/${ratio_name}"

    if [ -f "${out_dir}/sample.1.fq.gz" ] && [ -f "${out_dir}/sample.2.fq.gz" ]; then
        echo "Skipping ${sample_name} ${ratio_name} — already complete"
        continue
    fi

    echo "Coverage ratio: $ratio_name"
    mkdir -p "$out_dir"
    mkdir -p "$TMPDIR/reads"

    read -ra coverages <<< "${RATIO_SETS[$idx]}"

    # Simulate reads for each ref at its specific coverage
    for i in "${!coverages[@]}"; do
        ref_num=$((i + 1))
        ref_fa="$TMPDIR/ref${ref_num}.fa"
        cov="${coverages[$i]}"

        if [ ! -f "$ref_fa" ]; then
            echo "ERROR: Missing ref${ref_num}.fa for ${sample_name}"
            continue 2
        fi

        "$ART_BIN" -ss HS25 -na \
            -i "$ref_fa" \
            -f "$cov" \
            -l 100 -m 200 -s 10 \
            -o "$TMPDIR/reads/ref${ref_num}"

        if [ ! -f "$TMPDIR/reads/ref${ref_num}1.fq" ]; then
            echo "ERROR: ART failed for ref${ref_num} at ${cov}x"
            continue 2
        fi
    done

    # Combine reads from all refs
    cat "$TMPDIR"/reads/*1.fq > "$TMPDIR/sample.1.fq"
    cat "$TMPDIR"/reads/*2.fq > "$TMPDIR/sample.2.fq"

    background_fasta=$(ls "${BACKGROUND_DIR}"/*/contigs.fasta | shuf -n 1)

    "$ART_BIN" \
        -i "$background_fasta" \
        -f 10 \
        -l 100 -m 200 -s 10 \
        -o "$TMPDIR/background"

    if [ ! -f "$TMPDIR/background1.fq" ]; then
        echo "ERROR: ART background failed for ${sample_name} ${ratio_name}"
        rm -f "$TMPDIR/sample.1.fq" "$TMPDIR/sample.2.fq" "$TMPDIR"/reads/*.fq
        continue
    fi

    "$PYTHON_BIN" "$GEN_TITRATION" \
        -R1 "$TMPDIR/sample.1.fq" \
        -R2 "$TMPDIR/sample.2.fq" \
        -B1 "$TMPDIR/background1.fq" \
        -B2 "$TMPDIR/background2.fq" \
        --depth 3000000000 \
        -o "${out_dir}/sample"

    rm -f "$TMPDIR/sample.1.fq" "$TMPDIR/sample.2.fq" "$TMPDIR/background"*.fq "$TMPDIR"/reads/*.fq
    echo "Done: ${sample_name} ${ratio_name}"
done

echo "Finished ${sample_name} at $(date)"
