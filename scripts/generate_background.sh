#!/usr/bin/env bash
# =============================================================================
# generate_background.sh
#
# Builds the virus-free background metagenome pool using InSilicoSeq (ISS),
# simulating reads from a random subset of raw HumGut MAGs (prophages intact
# -- PHORAGER hasn't been run on HumGut yet; extract_humgut_subset.py pulls
# straight from the raw HumGut2.tar as a stand-in until that lands).
#
# Grid: config.ISS_MODELS x BACKGROUND_ABUNDANCE_DISTS x BACKGROUND_N_READS
# (2 x 2 x 1 = 4 combos), one SLURM array task per combo. Each MAG file is
# passed as its own --draft argument (ISS treats each --draft FILE as a single
# genome -- a multi-MAG file concatenated together would be wrongly modeled
# as one organism, verified empirically 2026-07-07).
#
# Read length is never hardcoded: after each run this script measures the
# actual R1 read length and read-pair count from the generated FASTQ and
# appends a row to config.BACKGROUND_MANIFEST. read_mixing.sh reads from that
# manifest, never from a per-model constant.
#
# Idempotent: skips a combo if its manifest row + output files already exist.
# The HumGut extraction step (extract_humgut_subset.py) is itself idempotent
# and flock-guarded so concurrent array tasks don't race on it.
#
# Usage:
#   sbatch --array=0-3 scripts/generate_background.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=generate_background
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
ISS_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/insilicoseq/bin/iss"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
HUMGUT_DIR="/rs1/researchers/b/blhurwit/users/nkolodi/databases/HumGut/raw_subset"
HUMGUT_GLOB="*.fasta"
EXTRACT_HUMGUT="$PROJECT/scripts/extract_humgut_subset.py"
BACKGROUND_DIR="$PROJECT/data/background_iss"
MANIFEST="$BACKGROUND_DIR/manifest.tsv"

mkdir -p "$PROJECT/logs/slurm" "$BACKGROUND_DIR" "$HUMGUT_DIR"

(
    flock -x 200
    if [ ! -s "$MANIFEST" ]; then
        printf "combo_name\tmodel\tabundance\tn_reads_requested\tn_pairs_actual\tread_length_actual\tseed\n" > "$MANIFEST"
    fi
) 200>"$MANIFEST.lock"

# --- Load the ART profile mapping from config.py (single source of truth; ---
# --- read_mixing.sh loads the same values the same way) ---------------------

declare -A ART_PROFILE_BY_MODEL=()
while IFS=$'\t' read -r k v; do ART_PROFILE_BY_MODEL["$k"]="$v"; done < <(
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT')
import config
for k, v in config.ART_PROFILE_BY_MODEL.items():
    print(f'{k}\t{v}')
"
)
declare -A ART_PROFILE_MAX_LEN=()
while IFS=$'\t' read -r k v; do ART_PROFILE_MAX_LEN["$k"]="$v"; done < <(
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT')
import config
for k, v in config.ART_PROFILE_MAX_LEN.items():
    print(f'{k}\t{v}')
"
)

# --- Prune manifest rows that can never be used by read_mixing.sh, and free -
# --- the disk space they occupy. A row is dead weight FOREVER (not just for
# --- this run's MODELS choice) if its model has no ART profile mapping, or
# --- its measured read length exceeds that profile's max -- ART fundamentally
# --- cannot produce spike-in reads at that length under ANY future MODELS
# --- setting (see read_mixing.sh's VALID_BG_ROWS, which applies the same
# --- test). A row for a model simply not in today's MODELS list, but still
# --- ART-compatible, is left alone since it could still be reused later.
# --- Flock-guarded with the same lock as the header-init/append blocks above
# --- so it's serialized against every concurrent array task; idempotent, so
# --- every task can safely attempt it -- only the first actually prunes.

(
    flock -x 200
    stale_combos=()
    keep_lines=("$(head -n1 "$MANIFEST")")
    while IFS=$'\t' read -r combo model abundance n_reads pairs read_len seed; do
        [ "$combo" = "combo_name" ] && continue
        [ -z "$combo" ] && continue
        profile="${ART_PROFILE_BY_MODEL[$model]:-}"
        profile_max="${ART_PROFILE_MAX_LEN[$profile]:-0}"
        if [ -n "$profile" ] && [ "$read_len" -le "$profile_max" ]; then
            keep_lines+=("$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s' \
                "$combo" "$model" "$abundance" "$n_reads" "$pairs" "$read_len" "$seed")")
        else
            stale_combos+=("$combo")
        fi
    done < <(tail -n +2 "$MANIFEST")

    if [ "${#stale_combos[@]}" -gt 0 ]; then
        printf '%s\n' "${keep_lines[@]}" > "$MANIFEST.tmp"
        mv "$MANIFEST.tmp" "$MANIFEST"
        for combo in "${stale_combos[@]}"; do
            echo "Pruned stale manifest row: $combo (incompatible with any ART profile) -- removing $BACKGROUND_DIR/$combo/"
            rm -rf "${BACKGROUND_DIR:?}/${combo:?}"
        done
    fi
) 200>"$MANIFEST.lock"

# --- Extract the raw HumGut MAG subset (idempotent, flock-guarded so the ---
# --- 8 concurrent array tasks don't race on the one-time tar extraction) ---

(
    flock -x 201
    "$PYTHON_BIN" "$EXTRACT_HUMGUT"
) 201>"$HUMGUT_DIR/.extract.lock"

mapfile -t MAG_FILES < <(compgen -G "$HUMGUT_DIR/$HUMGUT_GLOB" || true)
if [ "${#MAG_FILES[@]}" -eq 0 ]; then
    echo "ERROR: No HumGut MAG files found matching $HUMGUT_DIR/$HUMGUT_GLOB"
    echo "even after running $EXTRACT_HUMGUT. Check HumGut2.tar/HumGut2.tsv"
    echo "paths in config.py."
    exit 1
fi
echo "Found ${#MAG_FILES[@]} HumGut MAG files (raw, prophages intact)."

# --- Build the model x abundance x n_reads grid ------------------------------
# MODELS is loaded from config.ISS_MODELS (single source of truth) rather
# than hardcoded here -- currently (hiseq novaseq), the only two models whose
# ISS read length fits under some ART built-in profile (miseq/nextseq
# measure ~301bp, which exceeds every ART profile's cap of 250bp).

mapfile -t MODELS < <(
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT')
import config
for m in config.ISS_MODELS:
    print(m)
"
)
ABUNDANCES=(lognormal exponential)
N_READS_LIST=(1M)

combos=()
for model in "${MODELS[@]}"; do
    for abundance in "${ABUNDANCES[@]}"; do
        for n_reads in "${N_READS_LIST[@]}"; do
            combos+=("${model}:${abundance}:${n_reads}")
        done
    done
done
expected_combos=$(( ${#MODELS[@]} * ${#ABUNDANCES[@]} * ${#N_READS_LIST[@]} ))
echo "Total combos: ${#combos[@]} (expect $expected_combos, from ${#MODELS[@]} models x ${#ABUNDANCES[@]} abundances x ${#N_READS_LIST[@]} read depths)"

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

# --- Measure actual read length + pair count --

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
