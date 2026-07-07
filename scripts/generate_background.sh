#!/usr/bin/env bash
# =============================================================================
# generate_background.sh
#
# Builds the virus-free background metagenome pool using InSilicoSeq (ISS),
# simulating reads from PHORAGER's prophage-stripped HumGut MAGs.
#
# Grid: ISS_MODELS x ISS_ABUNDANCE_DISTS x BACKGROUND_N_READS (4 x 4 x 3 = 48
# combos), one SLURM array task per combo. Each MAG file is passed as its own
# --draft argument (ISS treats each --draft FILE as a single genome -- a
# multi-MAG file concatenated together would be wrongly modeled as one
# organism, verified empirically 2026-07-07).
#
# Read length is never hardcoded: after each run this script measures the
# actual R1 read length and read-pair count from the generated FASTQ and
# appends a row to config.BACKGROUND_MANIFEST. read_mixing.sh reads from that
# manifest, never from a per-model constant.
#
# Idempotent: skips a combo if its manifest row + output files already exist.
#
# Usage:
#   sbatch --array=0-47 scripts/generate_background.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=generate_background
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
ISS_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/insilicoseq/bin/iss"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
HUMGUT_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/databases/HumGut/prophage_removed"
HUMGUT_GLOB="*.fasta"
BACKGROUND_DIR="$PROJECT/data/background_iss"
MANIFEST="$BACKGROUND_DIR/manifest.tsv"

mkdir -p "$PROJECT/logs/slurm" "$BACKGROUND_DIR"

(
    flock -x 200
    if [ ! -s "$MANIFEST" ]; then
        printf "combo_name\tmodel\tabundance\tn_reads_requested\tn_pairs_actual\tread_length_actual\tseed\n" > "$MANIFEST"
    fi
) 200>"$MANIFEST.lock"

# --- Validate PHORAGER output exists -----------------------------------------

mapfile -t MAG_FILES < <(compgen -G "$HUMGUT_DIR/$HUMGUT_GLOB" || true)
if [ "${#MAG_FILES[@]}" -eq 0 ]; then
    echo "ERROR: No HumGut MAG files found matching $HUMGUT_DIR/$HUMGUT_GLOB"
    echo "PHORAGER has not been run on HumGut yet. This script needs one"
    echo "prophage-stripped MAG FASTA per file in that directory (see"
    echo "config.py: HUMGUT_PROPHAGE_REMOVED_DIR). Re-submit once PHORAGER"
    echo "output lands there."
    exit 1
fi
echo "Found ${#MAG_FILES[@]} HumGut MAG files."

# --- Build the model x abundance x n_reads grid ------------------------------

MODELS=(hiseq miseq novaseq nextseq)
ABUNDANCES=(lognormal halfnormal exponential uniform)
N_READS_LIST=(0.5M 1M 2M)

combos=()
for model in "${MODELS[@]}"; do
    for abundance in "${ABUNDANCES[@]}"; do
        for n_reads in "${N_READS_LIST[@]}"; do
            combos+=("${model}:${abundance}:${n_reads}")
        done
    done
done
echo "Total combos: ${#combos[@]} (expect 48)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: not running as a SLURM array job. Submit with:"
    echo "  sbatch --array=0-$(( ${#combos[@]} - 1 )) $0"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#combos[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of combos (${#combos[@]}). Nothing to do."
    exit 0
fi

IFS=':' read -r model abundance n_reads <<< "${combos[$SLURM_ARRAY_TASK_ID]}"
combo_name="${model}_${abundance}_${n_reads}"
out_dir="$BACKGROUND_DIR/$combo_name"
out_prefix="$out_dir/background"

echo "Combo: model=$model abundance=$abundance n_reads=$n_reads"

# --- Skip if already complete -------------------------------------------------

if [ -f "${out_prefix}_R1.fastq.gz" ] && [ -f "${out_prefix}_R2.fastq.gz" ] \
   && grep -qP "^${combo_name}\t" "$MANIFEST" 2>/dev/null; then
    echo "Skipping $combo_name -- already complete."
    exit 0
fi

mkdir -p "$out_dir"

seed=$(( 1000 + SLURM_ARRAY_TASK_ID ))

# --- Run ISS -------------------------------------------------------------------

"$ISS_BIN" generate \
    --draft "${MAG_FILES[@]}" \
    --model "$model" \
    --abundance "$abundance" \
    --n_reads "$n_reads" \
    --seed "$seed" \
    --cpus "$SLURM_CPUS_PER_TASK" \
    --compress \
    --output "$out_prefix"

if [ ! -f "${out_prefix}_R1.fastq.gz" ]; then
    echo "ERROR: ISS failed for $combo_name -- no output R1 produced."
    exit 1
fi

# --- Measure actual read length + pair count (never trust a hardcoded table) --

read_length=$(zcat "${out_prefix}_R1.fastq.gz" | awk 'NR==2{print length($0); exit}')
n_pairs_actual=$(zcat "${out_prefix}_R1.fastq.gz" | awk 'END{print NR/4}')

if [ -z "$read_length" ] || [ "$read_length" -le 0 ]; then
    echo "ERROR: could not measure read length for $combo_name"
    exit 1
fi

echo "Measured: read_length=$read_length n_pairs_actual=$n_pairs_actual"

# --- Append to manifest (flock-guarded: 48 array tasks write concurrently) ----

(
    flock -x 200
    if ! grep -qP "^${combo_name}\t" "$MANIFEST" 2>/dev/null; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$combo_name" "$model" "$abundance" "$n_reads" \
            "$n_pairs_actual" "$read_length" "$seed" >> "$MANIFEST"
    fi
) 200>"$MANIFEST.lock"

echo "Finished $combo_name at $(date)"
