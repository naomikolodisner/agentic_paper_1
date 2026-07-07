#!/usr/bin/env bash
# =============================================================================
# read_mixing.sh
#
# Builds the final spike-in FASTQ samples using InSilicoSeq (ISS), replacing
# the old ART-based pipeline entirely.
#
# Each sample dir (SPIKE_IN_DIR/sample_{size}v_{n}/) already has a fixed
# random subset of {size} viral references (contigs.fasta / refs_log.txt,
# written by pick_refs.py). For that fixed subset, this script builds every
# combination of:
#   - distribution in ISS_ABUNDANCE_DISTS (lognormal/halfnormal/exponential/uniform)
#     -- handed to ISS's own --abundance flag on the viral --genomes call, so
#     the subset's total read budget is split across its individual genomes
#     realistically (not evenly).
#   - percentage in SPIKE_PERCENTAGES (0.1%/1%/5%/10%)
#     -- the viral subset's total share of the final mixed sample's reads.
# (16 combos per sample dir: 4 distributions x 4 percentages.)
#
# For each combo:
#   1. Pick a random row from the background manifest (built by
#      generate_background.sh) -- gives a background model + actual read-pair
#      count + actual read length.
#   2. Compute the viral --n_reads target from that percentage via
#      viral_spike_helper.py total-reads.
#   3. iss generate --genomes contigs.fasta --model {same model as background}
#      --abundance {dist} --n_reads {n_reads} -- same model as the background
#      run so read lengths match by construction.
#   4. viral_spike_helper.py coverage-log -- derives per-genome coverage from
#      ISS's own abundance output (never an input).
#   5. combine_reads.py concatenates + shuffles the viral spike with the
#      chosen background into {dist}/spike_{pct}pct/final.{1,2}.fq.gz.
#
# Idempotent -- skips any dist/pct combo that already has both final FASTQs.
#
# Usage:
#   sbatch --array=0-14 scripts/read_mixing.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=read_mixing
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
ISS_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/insilicoseq/bin/iss"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
VIRAL_SPIKE_HELPER="$PROJECT/scripts/viral_spike_helper.py"
COMBINE_READS="$PROJECT/scripts/combine_reads.py"
SPIKE_IN_DIR="$PROJECT/data/spike_in_samples"
BACKGROUND_DIR="$PROJECT/data/background_iss"
BACKGROUND_MANIFEST="$BACKGROUND_DIR/manifest.tsv"

mkdir -p "$PROJECT/logs/slurm"

DISTRIBUTIONS=(lognormal halfnormal exponential uniform)
PERCENTAGES=(0.001 0.01 0.05 0.10)

# --- Validate prerequisites ---------------------------------------------------

if [ ! -f "$BACKGROUND_MANIFEST" ] || [ "$(wc -l < "$BACKGROUND_MANIFEST")" -le 1 ]; then
    echo "ERROR: $BACKGROUND_MANIFEST is missing or empty."
    echo "Run scripts/generate_background.sh first (it fills in this manifest)."
    exit 1
fi

# --- Select this array task's sample directory -------------------------------

mapfile -t SAMPLE_DIRS < <(find "$SPIKE_IN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'sample_*v_*' | sort)

if [ "${#SAMPLE_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: No sample_<size>v_<n> directories found in $SPIKE_IN_DIR."
    echo "Run scripts/pick_refs.py first."
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: not running as a SLURM array job. Submit with:"
    echo "  sbatch --array=0-$(( ${#SAMPLE_DIRS[@]} - 1 )) $0"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SAMPLE_DIRS[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of samples (${#SAMPLE_DIRS[@]}). Nothing to do."
    exit 0
fi

sample_dir="${SAMPLE_DIRS[$SLURM_ARRAY_TASK_ID]}"
sample_name=$(basename "$sample_dir")
echo "Processing: $sample_name"

contigs_fasta="${sample_dir}/contigs.fasta"
if [ ! -f "$contigs_fasta" ]; then
    echo "ERROR: $contigs_fasta not found. Run scripts/pick_refs.py first."
    exit 1
fi

# --- Setup temp directory ----------------------------------------------------

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

export OMP_NUM_THREADS=1

# =============================================================================
# Helper: build one dist/pct combo for the current sample
# =============================================================================
build_combo() {
    local dist="$1"
    local pct="$2"

    local pct_label
    pct_label=$(awk -v p="$pct" 'BEGIN{printf "%g", p*100}')
    local out_dir="${sample_dir}/${dist}/spike_${pct_label}pct"

    if [ -f "${out_dir}/final.1.fq.gz" ] && [ -f "${out_dir}/final.2.fq.gz" ]; then
        echo "Skipping $sample_name $dist ${pct_label}% -- already complete"
        return 0
    fi

    echo "Building $sample_name $dist ${pct_label}% ..."
    mkdir -p "$out_dir"

    local max_attempts=5
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))

        # Pick a random background manifest row (skip header)
        local bg_row
        bg_row=$(tail -n +2 "$BACKGROUND_MANIFEST" | shuf -n 1)
        if [ -z "$bg_row" ]; then
            echo "ERROR: could not pick a background row from $BACKGROUND_MANIFEST"
            return 1
        fi
        local bg_combo bg_model bg_abundance bg_n_reads bg_pairs bg_read_len bg_seed
        IFS=$'\t' read -r bg_combo bg_model bg_abundance bg_n_reads bg_pairs bg_read_len bg_seed <<< "$bg_row"
        local bg_prefix="$BACKGROUND_DIR/$bg_combo/background"

        if [ ! -f "${bg_prefix}_R1.fastq.gz" ] || [ ! -f "${bg_prefix}_R2.fastq.gz" ]; then
            echo "WARNING: background files missing for $bg_combo, retrying with a different row..."
            continue
        fi
        echo "  Background (attempt $attempt): $bg_combo (model=$bg_model, pairs=$bg_pairs, read_len=$bg_read_len)"

        local n_reads
        n_reads=$("$PYTHON_BIN" "$VIRAL_SPIKE_HELPER" total-reads \
            --background-pairs "$bg_pairs" --pct "$pct")

        local seed=$((RANDOM * RANDOM + attempt))
        local viral_prefix="$TMPDIR/viral_spike"
        rm -f "${viral_prefix}"*

        "$ISS_BIN" generate \
            --genomes "$contigs_fasta" \
            --model "$bg_model" \
            --abundance "$dist" \
            --n_reads "$n_reads" \
            --seed "$seed" \
            --cpus "$SLURM_CPUS_PER_TASK" \
            --output "$viral_prefix"

        if [ ! -f "${viral_prefix}_R1.fastq" ] || [ ! -f "${viral_prefix}_abundance.txt" ]; then
            echo "WARNING: ISS failed to produce viral spike reads, retrying..."
            continue
        fi

        "$PYTHON_BIN" "$VIRAL_SPIKE_HELPER" coverage-log \
            --abundance-file "${viral_prefix}_abundance.txt" \
            --contigs "$contigs_fasta" \
            --read-length "$bg_read_len" \
            --total-reads "$n_reads" \
            --out "${out_dir}/coverage_log.txt"

        "$PYTHON_BIN" "$COMBINE_READS" \
            -R1 "${viral_prefix}_R1.fastq" -R2 "${viral_prefix}_R2.fastq" \
            -B1 "${bg_prefix}_R1.fastq.gz" -B2 "${bg_prefix}_R2.fastq.gz" \
            --seed "$seed" \
            -o "$TMPDIR/final"

        if [ ! -f "$TMPDIR/final.1.fq.gz" ] || [ ! -f "$TMPDIR/final.2.fq.gz" ]; then
            echo "WARNING: combine_reads.py failed, retrying..."
            rm -f "$TMPDIR/final".*.fq.gz
            continue
        fi

        mv "$TMPDIR/final.1.fq.gz" "${out_dir}/final.1.fq.gz"
        mv "$TMPDIR/final.2.fq.gz" "${out_dir}/final.2.fq.gz"
        rm -f "${viral_prefix}"*
        echo "Done: $sample_name $dist ${pct_label}%"
        return 0
    done

    echo "ERROR: build_combo failed after $max_attempts attempts for $sample_name $dist ${pct_label}%"
    return 1
}

# =============================================================================
# Build every distribution x percentage combo for this sample
# =============================================================================
overall_status=0
for dist in "${DISTRIBUTIONS[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
        build_combo "$dist" "$pct" || overall_status=1
    done
done

echo "Finished $sample_name at $(date)"
exit $overall_status
