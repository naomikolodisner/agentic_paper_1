#!/usr/bin/env bash
# =============================================================================
# read_mixing.sh
#
# Builds the final spike-in FASTQ samples using ART Illumina for the viral
# reads, mixed with the FMT-profile-driven InSilicoSeq (ISS) background pool
# built by generate_background_fmt.sh. Replicates the PI's original ART
# scripts (scripts/archive/create_single_equal_samples.sh,
# create_unequal_samples_slurm.sh): each genome is simulated at a FIXED
# absolute ART fold coverage (-f), and the viral read percentage is a
# derived, logged quantity -- never an input.
#
# Every sample dir is one that pick_refs.py wrote a conditions.tsv into:
#   --design og:      SPIKE_IN_DIR/{single,equal2,...,unequal3}/sample{1..15}/
#   --design subsets: SPIKE_IN_DIR/sample_{size}v_{n}/
# Each sample dir has contigs.fasta (its fixed genome subset), refs_log.txt,
# and conditions.tsv: one row per condition, condition_name<TAB>comma-separated
# per-genome coverages in contigs.fasta order (e.g. "1x_0.5x<TAB>1,0.5").
#
# For each condition (one output dir per condition, like the original):
#   1. Draw ONE random usable background row (VALID_BG_ROWS below) -- like the
#      original, which drew a random background per coverage condition.
#   2. art_illumina -p (paired-end) per genome at that genome's coverage,
#      read length matched to the background's measured length,
#      concatenated into one viral R1/R2 pair.
#   3. viral_spike_helper.py log-actual-coverage -- per-genome coverage
#      derived from ART's real per-genome read counts.
#   4. spike_summary.tsv -- chosen background, viral pairs, derived viral %.
#   5. combine_reads.py concatenates + shuffles the viral spike with the
#      background into {sample_dir}/{condition}/final.{1,2}.fq.gz.
#
# Idempotent -- skips any condition that already has both final FASTQs.
# Retries (up to 5x, fresh ART seed each time) reuse the SAME background row.
#
# Usage (array size = number of sample dirs with a conditions.tsv):
#   N=$(find data/spike_in_samples -name conditions.tsv | wc -l)
#   sbatch --array=0-$((N - 1)) scripts/read_mixing.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=read_mixing
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
BACKGROUND_DIR="$PROJECT/data/background_fmt"  # config.OUT_BACKGROUND_FMT
BACKGROUND_MANIFEST="$BACKGROUND_DIR/manifest.tsv"

mkdir -p "$PROJECT/logs/slurm"

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
    echo "Run scripts/generate_background_fmt.sh first (it fills in this manifest)."
    exit 1
fi

# --- Pre-filter the manifest to rows an ART profile can actually simulate ----
# --- spike-in reads at (matching bg_read_len), so the random draw below  ---
# --- always picks a row that is guaranteed to work.                      ---

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
    echo "mapped ART profile (ART_PROFILE_BY_MODEL / ART_PROFILE_MAX_LEN in config.py)."
    echo "Run scripts/generate_background_fmt.sh with a compatible model."
    exit 1
fi
echo "Usable background rows: ${#VALID_BG_ROWS[@]} (of $(( $(wc -l < "$BACKGROUND_MANIFEST") - 1 )) total in manifest)"

# --- Select this array task's sample directory -------------------------------

mapfile -t SAMPLE_DIRS < <(find "$SPIKE_IN_DIR" -name conditions.tsv -printf '%h\n' | sort)

if [ "${#SAMPLE_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: No sample dirs with a conditions.tsv found under $SPIKE_IN_DIR."
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
sample_name="${sample_dir#$SPIKE_IN_DIR/}"
echo "Processing: $sample_name"

contigs_fasta="${sample_dir}/contigs.fasta"
conditions_tsv="${sample_dir}/conditions.tsv"
if [ ! -f "$contigs_fasta" ]; then
    echo "ERROR: $contigs_fasta not found. Run scripts/pick_refs.py first."
    exit 1
fi

# Genome IDs in contigs.fasta order -- conditions.tsv coverages line up with these
mapfile -t GENOME_IDS < <(grep '^>' "$contigs_fasta" | sed 's/^>//; s/[[:space:]].*//')

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
# Helper: build one condition for the current sample
# =============================================================================
build_condition() {
    local cond="$1"
    local cov_csv="$2"

    local -a covs
    IFS=',' read -ra covs <<< "$cov_csv"
    if [ "${#covs[@]}" -ne "${#GENOME_IDS[@]}" ]; then
        echo "ERROR: condition $cond has ${#covs[@]} coverages but contigs.fasta has ${#GENOME_IDS[@]} genomes"
        return 1
    fi

    local out_dir="${sample_dir}/${cond}"

    if [ -f "${out_dir}/final.1.fq.gz" ] && [ -f "${out_dir}/final.2.fq.gz" ]; then
        echo "Skipping $sample_name $cond -- already complete"
        return 0
    fi

    # One random background per condition, like the original scripts
    local bg_row
    bg_row="${VALID_BG_ROWS[$(( RANDOM % ${#VALID_BG_ROWS[@]} ))]}"

    local bg_combo bg_model bg_abundance bg_n_reads bg_pairs bg_read_len bg_seed
    IFS=$'\t' read -r bg_combo bg_model bg_abundance bg_n_reads bg_pairs bg_read_len bg_seed <<< "$bg_row"
    local bg_prefix="$BACKGROUND_DIR/$bg_combo/background"

    if [ ! -f "${bg_prefix}_R1.fastq.gz" ] || [ ! -f "${bg_prefix}_R2.fastq.gz" ]; then
        echo "ERROR: background files missing for $bg_combo -- skipping $sample_name $cond"
        return 1
    fi

    local art_profile="${ART_PROFILE_BY_MODEL[$bg_model]}"
    echo "Building $sample_name $cond (coverages=$cov_csv, bg=$bg_combo, pairs=$bg_pairs, read_len=$bg_read_len, art_profile=$art_profile) ..."
    mkdir -p "$out_dir"

    local max_attempts=5
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))

        local seed=$((RANDOM * RANDOM + attempt))

        local reads_dir="$TMPDIR/reads_${cond}"
        rm -rf "$reads_dir"
        mkdir -p "$reads_dir"

        local actual_pairs_file="$TMPDIR/actual_pairs_${cond}.tsv"
        : > "$actual_pairs_file"

        local art_failed=0
        local i genome_id cov
        for i in "${!GENOME_IDS[@]}"; do
            genome_id="${GENOME_IDS[$i]}"
            cov="${covs[$i]}"

            "$ART_BIN" -ss "$art_profile" -p -na \
                -i "$TMPDIR/refs/${genome_id}.fa" \
                -f "$cov" \
                -l "$bg_read_len" \
                -m "$ART_FRAGMENT_MEAN" -s "$ART_FRAGMENT_SD" \
                -rs "$seed" \
                -o "${reads_dir}/${genome_id}_"

            if [ ! -f "${reads_dir}/${genome_id}_1.fq" ]; then
                echo "WARNING: ART failed for $genome_id at ${cov}x"
                art_failed=1
                break
            fi

            local n_pairs_actual
            n_pairs_actual=$(awk 'END{print NR/4}' "${reads_dir}/${genome_id}_1.fq")
            printf "%s\t%s\n" "$genome_id" "$n_pairs_actual" >> "$actual_pairs_file"
        done

        if [ "$art_failed" -eq 1 ]; then
            echo "WARNING: ART step failed (attempt $attempt), retrying..."
            continue
        fi

        cat "${reads_dir}"/*_1.fq > "$TMPDIR/viral_${cond}.1.fq" 2>/dev/null
        cat "${reads_dir}"/*_2.fq > "$TMPDIR/viral_${cond}.2.fq" 2>/dev/null

        if [ ! -s "$TMPDIR/viral_${cond}.1.fq" ]; then
            echo "WARNING: no viral reads produced (attempt $attempt), retrying..."
            continue
        fi

        "$PYTHON_BIN" "$VIRAL_SPIKE_HELPER" log-actual-coverage \
            --contigs "$contigs_fasta" \
            --actual-pairs-file "$actual_pairs_file" \
            --read-length "$bg_read_len" \
            --out "${out_dir}/coverage_log.txt"

        # Derived viral fraction of the final sample (logged, never an input)
        awk -F'\t' -v cond="$cond" -v covs="$cov_csv" -v bg="$bg_combo" -v bgp="$bg_pairs" -v seed="$seed" '
            {v += $2}
            END {
                print "condition\tcoverages\tbg_combo\tviral_pairs\tbg_pairs\tviral_pct\tseed"
                printf "%s\t%s\t%s\t%d\t%d\t%.4f\t%s\n", cond, covs, bg, v, bgp, 100 * v / (v + bgp), seed
            }' "$actual_pairs_file" > "${out_dir}/spike_summary.tsv"

        "$PYTHON_BIN" "$COMBINE_READS" \
            -R1 "$TMPDIR/viral_${cond}.1.fq" -R2 "$TMPDIR/viral_${cond}.2.fq" \
            -B1 "${bg_prefix}_R1.fastq.gz" -B2 "${bg_prefix}_R2.fastq.gz" \
            --seed "$seed" \
            -o "$TMPDIR/final_${cond}"

        if [ ! -f "$TMPDIR/final_${cond}.1.fq.gz" ] || [ ! -f "$TMPDIR/final_${cond}.2.fq.gz" ]; then
            echo "WARNING: combine_reads.py failed (attempt $attempt), retrying..."
            rm -f "$TMPDIR/final_${cond}".*.fq.gz
            continue
        fi

        mv "$TMPDIR/final_${cond}.1.fq.gz" "${out_dir}/final.1.fq.gz"
        mv "$TMPDIR/final_${cond}.2.fq.gz" "${out_dir}/final.2.fq.gz"
        rm -rf "$reads_dir"
        rm -f "$TMPDIR/viral_${cond}".*.fq "$actual_pairs_file"
        echo "Done: $sample_name $cond"
        return 0
    done

    echo "ERROR: build_condition failed after $max_attempts attempts for $sample_name $cond"
    return 1
}

# =============================================================================
# Build every condition in this sample's conditions.tsv
# =============================================================================
overall_status=0
while IFS=$'\t' read -r -u 3 cond cov_csv; do
    [ -z "$cond" ] && continue
    build_condition "$cond" "$cov_csv" || overall_status=1
done 3< "$conditions_tsv"

echo "Finished $sample_name at $(date)"
exit $overall_status
