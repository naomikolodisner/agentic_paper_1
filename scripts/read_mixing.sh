#!/usr/bin/env bash
# =============================================================================
# read_mixing.sh
#
# Simulates Illumina reads from viral spike-in sequences and mixes them with a
# randomly selected background metagenome to produce synthetic paired-end FASTQ.
#
# Works for ALL sample types: single, equal2, equal3, equal4, unequal2, unequal3
# The sample type is detected automatically from the name of SAMPLES_DIR.
#
# Equal samples (single / equal2 / equal3 / equal4):
#   All viral references in contigs.fasta are simulated together at 4 uniform
#   coverages: 0.1x, 0.5x, 1x, 10x. Output goes to {sample}/{cov}x/.
#
# Unequal samples (unequal2 / unequal3):
#   Each viral reference is simulated at a different coverage so the resulting
#   mixture has unequal virus abundances:
#     unequal2 → 10x:1x  and  1x:0.5x
#     unequal3 → 10x:1x:0.5x  and  1x:0.5x:0.1x
#   Output goes to {sample}/{ratio_name}/.
#
# The script is idempotent — it skips any coverage/ratio directory that already
# has both output FASTQ files.
#
# Prerequisites:
#   pick_refs.py must have been run first so that each sample directory contains
#   a contigs.fasta with the viral reference sequences to spike in.
#
# Usage (submit one sample type at a time):
#   sbatch --array=0-14 --export=ALL,SAMPLES_DIR=.../spike_in_samples/single   read_mixing.sh
#   sbatch --array=0-14 --export=ALL,SAMPLES_DIR=.../spike_in_samples/equal2   read_mixing.sh
#   sbatch --array=0-14 --export=ALL,SAMPLES_DIR=.../spike_in_samples/equal3   read_mixing.sh
#   sbatch --array=0-14 --export=ALL,SAMPLES_DIR=.../spike_in_samples/equal4   read_mixing.sh
#   sbatch --array=0-14 --export=ALL,SAMPLES_DIR=.../spike_in_samples/unequal2 read_mixing.sh
#   sbatch --array=0-14 --export=ALL,SAMPLES_DIR=.../spike_in_samples/unequal3 read_mixing.sh
#
# Or just run submit_pipeline.sh to submit all types at once.
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=read_mixing
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
GEN_TITRATION="$PROJECT/scripts/gen_titration_sample.py"
BACKGROUND_DIR="$PROJECT/data/no_virus_contigs"
ART_BIN="$PROJECT/../conda_envs/test_env/bin/art_illumina"
PYTHON_BIN="$PROJECT/../conda_envs/test_env/bin/python"

mkdir -p "$PROJECT/logs/slurm"

# --- Validate inputs ---------------------------------------------------------

if [ -z "$SAMPLES_DIR" ]; then
    echo "ERROR: SAMPLES_DIR is not set."
    echo "Submit with: sbatch --array=0-14 --export=ALL,SAMPLES_DIR=/path/to/sample_type read_mixing.sh"
    exit 1
fi

if [ ! -f "$ART_BIN" ]; then
    echo "ERROR: art_illumina not found at $ART_BIN"
    exit 1
fi

# --- Select this array task's sample directory -------------------------------

mapfile -t SAMPLE_DIRS < <(find "$SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SAMPLE_DIRS[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of samples (${#SAMPLE_DIRS[@]}). Nothing to do."
    exit 0
fi

sample_dir="${SAMPLE_DIRS[$SLURM_ARRAY_TASK_ID]}"
sample_name=$(basename "$sample_dir")
sample_type=$(basename "$SAMPLES_DIR")   # e.g. "single", "equal2", "unequal3"

echo "Sample type: $sample_type"
echo "Processing:  $sample_name"

# --- Setup temp directory ----------------------------------------------------

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

export OMP_NUM_THREADS=1

# --- Check that pick_refs.py has been run ------------------------------------

contigs_fasta="${sample_dir}/contigs.fasta"
if [ ! -f "$contigs_fasta" ]; then
    echo "ERROR: $contigs_fasta not found."
    echo "Run pick_refs.py first to generate viral reference sequences for each sample."
    exit 1
fi

# =============================================================================
# Helper: simulate background reads and mix with viral reads
# Arguments: $1 = output directory, $2 = viral R1, $3 = viral R2
# =============================================================================
mix_with_background() {
    local out_dir="$1"
    local viral_r1="$2"
    local viral_r2="$3"

    local max_attempts=5
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))

        local background_fasta
        background_fasta=$(ls "${BACKGROUND_DIR}"/*/contigs.fasta | shuf -n 1)
        echo "  Background fasta (attempt $attempt): $(basename $(dirname $background_fasta))"

        "$ART_BIN" -na \
            -i "$background_fasta" \
            -f 10 \
            -l 100 -m 200 -s 10 \
            -o "$TMPDIR/background"

        if [ ! -f "$TMPDIR/background1.fq" ] || [ ! -s "$TMPDIR/background1.fq" ]; then
            echo "WARNING: ART produced empty/missing background1.fq, retrying..."
            rm -f "$TMPDIR/background"*.fq
            continue
        fi
        if [ ! -f "$TMPDIR/background2.fq" ] || [ ! -s "$TMPDIR/background2.fq" ]; then
            echo "WARNING: ART produced empty/missing background2.fq, retrying..."
            rm -f "$TMPDIR/background"*.fq
            continue
        fi

        "$PYTHON_BIN" "$GEN_TITRATION" \
            -R1 "$viral_r1" \
            -R2 "$viral_r2" \
            -B1 "$TMPDIR/background1.fq" \
            -B2 "$TMPDIR/background2.fq" \
            --depth 3000000000 \
            -o "${out_dir}/sample"

        local exit_code=$?
        rm -f "$TMPDIR/background"*.fq

        if [ $exit_code -eq 0 ]; then
            return 0
        fi
        echo "WARNING: gen_titration_sample.py failed (exit $exit_code), retrying with different background..."
        rm -f "${out_dir}/sample"*.fq.gz
    done

    echo "ERROR: mix_with_background failed after $max_attempts attempts for $out_dir"
    exit 1
}

# =============================================================================
# EQUAL samples: single, equal2, equal3, equal4
#   All refs simulated together at uniform coverage.
# =============================================================================
run_equal() {
    local coverages=(0.1 0.5 1 10)

    for cov in "${coverages[@]}"; do
        local out_dir="${sample_dir}/${cov}x"

        if [ -f "${out_dir}/sample.1.fq.gz" ] && [ -f "${out_dir}/sample.2.fq.gz" ]; then
            echo "Skipping $sample_name ${cov}x — already complete"
            continue
        fi

        echo "Simulating ${sample_name} at ${cov}x coverage..."
        mkdir -p "$out_dir"

        "$ART_BIN" -ss HS25 -na \
            -i "$contigs_fasta" \
            -f "$cov" \
            -l 100 -m 200 -s 10 \
            -o "$TMPDIR/sample"

        if [ ! -f "$TMPDIR/sample1.fq" ]; then
            echo "ERROR: ART failed for ${sample_name} at ${cov}x"
            exit 1
        fi

        mv "$TMPDIR/sample1.fq" "$TMPDIR/sample.1.fq"
        mv "$TMPDIR/sample2.fq" "$TMPDIR/sample.2.fq"

        mix_with_background "$out_dir" "$TMPDIR/sample.1.fq" "$TMPDIR/sample.2.fq"
        rm -f "$TMPDIR/sample".*.fq

        echo "Done: $sample_name ${cov}x"
    done
}

# =============================================================================
# UNEQUAL samples: unequal2, unequal3
#   Each ref simulated separately at a different coverage.
# =============================================================================
run_unequal() {
    local num_refs="$1"

    # Define coverage ratios for each configuration
    local ratio_names=()
    local ratio_sets=()
    if [ "$num_refs" -eq 2 ]; then
        ratio_names=("10x_1x" "1x_0.5x")
        ratio_sets=("10 1" "1 0.5")
    elif [ "$num_refs" -eq 3 ]; then
        ratio_names=("10x_1x_0.5x" "1x_0.5x_0.1x")
        ratio_sets=("10 1 0.5" "1 0.5 0.1")
    else
        echo "ERROR: unrecognized unequal type (expected 2 or 3, got $num_refs)"
        exit 1
    fi

    # Split contigs.fasta into one file per viral reference sequence
    awk '/^>/{f=ENVIRON["TMPDIR"]"/ref"(++i)".fa"} {print > f}' "$contigs_fasta"

    mkdir -p "$TMPDIR/reads"

    for idx in "${!ratio_names[@]}"; do
        local ratio_name="${ratio_names[$idx]}"
        local out_dir="${sample_dir}/${ratio_name}"

        if [ -f "${out_dir}/sample.1.fq.gz" ] && [ -f "${out_dir}/sample.2.fq.gz" ]; then
            echo "Skipping $sample_name $ratio_name — already complete"
            continue
        fi

        echo "Simulating ${sample_name} at ratio ${ratio_name}..."
        mkdir -p "$out_dir"

        read -ra coverages <<< "${ratio_sets[$idx]}"

        for i in "${!coverages[@]}"; do
            local ref_num=$((i + 1))
            local ref_fa="$TMPDIR/ref${ref_num}.fa"
            local cov="${coverages[$i]}"

            if [ ! -f "$ref_fa" ]; then
                echo "ERROR: Expected ref${ref_num}.fa but it was not found in $TMPDIR"
                exit 1
            fi

            "$ART_BIN" -ss HS25 -na \
                -i "$ref_fa" \
                -f "$cov" \
                -l 100 -m 200 -s 10 \
                -o "$TMPDIR/reads/ref${ref_num}"

            if [ ! -f "$TMPDIR/reads/ref${ref_num}1.fq" ]; then
                echo "ERROR: ART failed for ref${ref_num} at ${cov}x"
                exit 1
            fi
        done

        cat "$TMPDIR"/reads/*1.fq > "$TMPDIR/sample.1.fq"
        cat "$TMPDIR"/reads/*2.fq > "$TMPDIR/sample.2.fq"

        mix_with_background "$out_dir" "$TMPDIR/sample.1.fq" "$TMPDIR/sample.2.fq"
        rm -f "$TMPDIR/sample".*.fq "$TMPDIR/reads/"*.fq

        echo "Done: $sample_name $ratio_name"
    done
}

# =============================================================================
# Dispatch based on sample type
# =============================================================================
if [[ "$sample_type" == unequal* ]]; then
    num_refs="${sample_type#unequal}"
    run_unequal "$num_refs"
else
    run_equal
fi

echo "Finished $sample_name at $(date)"
