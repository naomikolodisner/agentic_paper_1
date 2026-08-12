#!/usr/bin/env bash
# =============================================================================
# read_mixing.sh
#
# Builds the final spike-in FASTQ samples using ART Illumina for the viral
# reads, mixed with the InSilicoSeq (ISS) background pool built by
# generate_background.sh. This mirrors the archived
# equal2/equal3/equal4/unequal2/unequal3 scripts (one script, parametrized,
# handles every subset size) but generalizes their fixed per-genome coverage
# values into ART_RATIO_WEIGHTS -- relative weights cycled across an
# arbitrary-size genome subset -- and drives the TOTAL viral read budget from
# a percentage-of-final-reads target instead of fixed absolute coverages.
#
# Each sample dir (SPIKE_IN_DIR/sample_{size}v_{n}/) already has a fixed
# random subset of {size} viral references (contigs.fasta / refs_log.txt,
# written by pick_refs.py). For that fixed subset, this script builds EVERY
# combination of percentage in config.SPIKE_PERCENTAGES (1%/5%/10%) x every
# background row an ART profile can actually simulate spike-in reads at
# (VALID_BG_ROWS below) -- exhaustive, not a random per-combo draw. With 3
# percentages x 4 usable background rows (hiseq/novaseq x lognormal/
# exponential), that's 12 combos per sample dir, 36 across all 3 dirs.
#
# For each (percentage, background row) combo:
#   1. viral_spike_helper.py total-reads -- computes the total viral read
#      target from that percentage and the background row's actual read-pair
#      count.
#   2. viral_spike_helper.py weighted-coverage-plan -- splits that total
#      UNEVENLY across the subset's genomes by cycling ART_RATIO_WEIGHTS
#      (10:1:0.5:0.1, the old scripts' fixed coverages reused as relative
#      weights), converting each genome's share into an ART -f fold-coverage
#      target via its own real length.
#   3. art_illumina -p (paired-end) per genome at its planned coverage,
#      concatenated into one viral R1/R2 pair.
#   4. viral_spike_helper.py log-actual-coverage -- derives per-genome
#      coverage from ART's real per-genome read counts (never the -f input).
#   5. combine_reads.py concatenates + shuffles the viral spike with this
#      background row into spike_{pct}pct_{bg_combo}/final.{1,2}.fq.gz.
#
# Idempotent -- skips any (pct, bg_row) combo that already has both final
# FASTQs. Retries (up to 5x, fresh ART seed each time) are against the SAME
# background row -- the row is no longer randomly redrawn on failure, since
# every row is now built deliberately rather than picked.
#
# Usage:
#   sbatch --array=0-2 scripts/read_mixing.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=read_mixing
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Node: $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
ART_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/art_illumina"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
VIRAL_SPIKE_HELPER="$PROJECT/scripts/viral_spike_helper.py"
COMBINE_READS="$PROJECT/scripts/combine_reads.py"
SPIKE_IN_DIR="$PROJECT/data/spike_in_samples"
BACKGROUND_DIR="$PROJECT/data/background_iss"
BACKGROUND_MANIFEST="$BACKGROUND_DIR/manifest.tsv"

mkdir -p "$PROJECT/logs/slurm"

# Loaded from config.py (single source of truth) rather than hardcoded here.
mapfile -t PERCENTAGES < <(
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT')
import config
for p in config.SPIKE_PERCENTAGES:
    print(p)
"
)

ART_FRAGMENT_MEAN=200
ART_FRAGMENT_SD=10

# Loaded from config.py (single source of truth; generate_background.sh's
# manifest-pruning step loads the same values the same way) rather than
# hardcoded here -- a prior mismatch (novaseq mapped to a profile 1bp too
# short) went unnoticed for a while because this used to be a hand-maintained
# duplicate of the Python dict.
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

# --- Validate prerequisites ---------------------------------------------------

if [ ! -f "$BACKGROUND_MANIFEST" ] || [ "$(wc -l < "$BACKGROUND_MANIFEST")" -le 1 ]; then
    echo "ERROR: $BACKGROUND_MANIFEST is missing or empty."
    echo "Run scripts/generate_background.sh first (it fills in this manifest)."
    exit 1
fi

# --- Pre-filter the manifest to rows an ART profile can actually simulate ----
# --- spike-in reads at (matching bg_read_len). generate_background.sh is  ---
# --- additive/idempotent across reruns with different MODELS, so stale   ---
# --- rows from an earlier model choice (e.g. miseq at 301bp, which no    ---
# --- ART built-in profile supports -- max is 250bp) can still be sitting ---
# --- in the manifest. Filtering here means shuf always draws a row that  ---
# --- is guaranteed to work, instead of randomly retrying into a row that ---
# --- can NEVER work and burning through max_attempts on bad luck alone.  ---

mapfile -t VALID_BG_ROWS < <(
    tail -n +2 "$BACKGROUND_MANIFEST" | while IFS=$'\t' read -r combo model abundance n_reads pairs read_len seed; do
        profile="${ART_PROFILE_BY_MODEL[$model]:-}"
        [ -z "$profile" ] && continue
        profile_max="${ART_PROFILE_MAX_LEN[$profile]:-0}"
        [ "$read_len" -le "$profile_max" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$combo" "$model" "$abundance" "$n_reads" "$pairs" "$read_len" "$seed"
    done
)

if [ "${#VALID_BG_ROWS[@]}" -eq 0 ]; then
    echo "ERROR: no row in $BACKGROUND_MANIFEST has a read length compatible with any"
    echo "mapped ART profile (ART_PROFILE_BY_MODEL / ART_PROFILE_MAX_LEN in this script)."
    echo "Run scripts/generate_background.sh with a compatible model (e.g. hiseq)."
    exit 1
fi
echo "Usable background rows: ${#VALID_BG_ROWS[@]} (of $(( $(wc -l < "$BACKGROUND_MANIFEST") - 1 )) total in manifest)"

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

# --- Split contigs.fasta into one file per genome, named by its real record ---
# --- ID (matches genome_id as read by viral_spike_helper.py's SeqIO.parse) ---

mkdir -p "$TMPDIR/refs"
"$PYTHON_BIN" - "$contigs_fasta" "$TMPDIR/refs" <<'EOF'
import sys
from Bio import SeqIO

contigs, out_dir = sys.argv[1], sys.argv[2]
for record in SeqIO.parse(contigs, "fasta"):
    SeqIO.write(record, f"{out_dir}/{record.id}.fa", "fasta")
EOF

# =============================================================================
# Helper: build one (percentage, background row) combo for the current sample
# =============================================================================
build_combo() {
    local pct="$1"
    local bg_row="$2"

    local pct_label
    pct_label=$(awk -v p="$pct" 'BEGIN{printf "%g", p*100}')

    local bg_combo bg_model bg_abundance bg_n_reads bg_pairs bg_read_len bg_seed
    IFS=$'\t' read -r bg_combo bg_model bg_abundance bg_n_reads bg_pairs bg_read_len bg_seed <<< "$bg_row"
    local bg_prefix="$BACKGROUND_DIR/$bg_combo/background"
    local combo_label="${pct_label}pct_${bg_combo}"
    local out_dir="${sample_dir}/spike_${combo_label}"

    if [ -f "${out_dir}/final.1.fq.gz" ] && [ -f "${out_dir}/final.2.fq.gz" ]; then
        echo "Skipping $sample_name ${pct_label}% x $bg_combo -- already complete"
        return 0
    fi

    if [ ! -f "${bg_prefix}_R1.fastq.gz" ] || [ ! -f "${bg_prefix}_R2.fastq.gz" ]; then
        echo "ERROR: background files missing for $bg_combo -- skipping $sample_name ${pct_label}% x $bg_combo"
        return 1
    fi

    local art_profile="${ART_PROFILE_BY_MODEL[$bg_model]}"
    echo "Building $sample_name ${pct_label}% x $bg_combo (model=$bg_model, pairs=$bg_pairs, read_len=$bg_read_len, art_profile=$art_profile) ..."
    mkdir -p "$out_dir"

    local max_attempts=5
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))

        local n_reads
        n_reads=$("$PYTHON_BIN" "$VIRAL_SPIKE_HELPER" total-reads \
            --background-pairs "$bg_pairs" --pct "$pct")

        local seed=$((RANDOM * RANDOM + attempt))

        local plan_tsv="$TMPDIR/plan_${combo_label}.tsv"
        "$PYTHON_BIN" "$VIRAL_SPIKE_HELPER" weighted-coverage-plan \
            --contigs "$contigs_fasta" \
            --total-reads "$n_reads" \
            --read-length "$bg_read_len" \
            --out "$plan_tsv"

        if [ ! -s "$plan_tsv" ]; then
            echo "WARNING: weighted-coverage-plan produced no output (attempt $attempt), retrying..."
            continue
        fi

        local reads_dir="$TMPDIR/reads_${combo_label}"
        rm -rf "$reads_dir"
        mkdir -p "$reads_dir"

        local actual_pairs_file="$TMPDIR/actual_pairs_${combo_label}.tsv"
        : > "$actual_pairs_file"

        local art_failed=0
        while IFS=$'\t' read -r genome_id genome_length weight n_pairs_target coverage_target; do
            [ "$genome_id" = "genome_id" ] && continue  # skip header

            local ref_fa="$TMPDIR/refs/${genome_id}.fa"
            if [ ! -f "$ref_fa" ]; then
                echo "WARNING: missing split ref FASTA for $genome_id"
                art_failed=1
                break
            fi

            "$ART_BIN" -ss "$art_profile" -p -na \
                -i "$ref_fa" \
                -f "$coverage_target" \
                -l "$bg_read_len" \
                -m "$ART_FRAGMENT_MEAN" -s "$ART_FRAGMENT_SD" \
                -rs "$seed" \
                -o "${reads_dir}/${genome_id}_"

            if [ ! -f "${reads_dir}/${genome_id}_1.fq" ]; then
                echo "WARNING: ART failed for $genome_id at ${coverage_target}x"
                art_failed=1
                break
            fi

            local n_pairs_actual
            n_pairs_actual=$(awk 'END{print NR/4}' "${reads_dir}/${genome_id}_1.fq")
            printf "%s\t%s\n" "$genome_id" "$n_pairs_actual" >> "$actual_pairs_file"
        done < "$plan_tsv"

        if [ "$art_failed" -eq 1 ]; then
            echo "WARNING: ART step failed (attempt $attempt), retrying..."
            continue
        fi

        cat "${reads_dir}"/*_1.fq > "$TMPDIR/viral_${combo_label}.1.fq" 2>/dev/null
        cat "${reads_dir}"/*_2.fq > "$TMPDIR/viral_${combo_label}.2.fq" 2>/dev/null

        if [ ! -s "$TMPDIR/viral_${combo_label}.1.fq" ]; then
            echo "WARNING: no viral reads produced (attempt $attempt), retrying..."
            continue
        fi

        "$PYTHON_BIN" "$VIRAL_SPIKE_HELPER" log-actual-coverage \
            --contigs "$contigs_fasta" \
            --actual-pairs-file "$actual_pairs_file" \
            --read-length "$bg_read_len" \
            --out "${out_dir}/coverage_log.txt"

        "$PYTHON_BIN" "$COMBINE_READS" \
            -R1 "$TMPDIR/viral_${combo_label}.1.fq" -R2 "$TMPDIR/viral_${combo_label}.2.fq" \
            -B1 "${bg_prefix}_R1.fastq.gz" -B2 "${bg_prefix}_R2.fastq.gz" \
            --seed "$seed" \
            -o "$TMPDIR/final_${combo_label}"

        if [ ! -f "$TMPDIR/final_${combo_label}.1.fq.gz" ] || [ ! -f "$TMPDIR/final_${combo_label}.2.fq.gz" ]; then
            echo "WARNING: combine_reads.py failed (attempt $attempt), retrying..."
            rm -f "$TMPDIR/final_${combo_label}".*.fq.gz
            continue
        fi

        mv "$TMPDIR/final_${combo_label}.1.fq.gz" "${out_dir}/final.1.fq.gz"
        mv "$TMPDIR/final_${combo_label}.2.fq.gz" "${out_dir}/final.2.fq.gz"
        rm -rf "$reads_dir"
        rm -f "$TMPDIR/viral_${combo_label}".*.fq "$plan_tsv" "$actual_pairs_file"
        echo "Done: $sample_name ${pct_label}% x $bg_combo"
        return 0
    done

    echo "ERROR: build_combo failed after $max_attempts attempts for $sample_name ${pct_label}% x $bg_combo"
    return 1
}

# =============================================================================
# Build every (percentage x background row) combo for this sample --
# exhaustive, not a random per-combo pick.
# =============================================================================
overall_status=0
for pct in "${PERCENTAGES[@]}"; do
    for bg_row in "${VALID_BG_ROWS[@]}"; do
        build_combo "$pct" "$bg_row" || overall_status=1
    done
done

echo "Finished $sample_name at $(date)"
exit $overall_status
