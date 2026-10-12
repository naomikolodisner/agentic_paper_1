#!/usr/bin/env bash
# =============================================================================
# generate_background_fmt.sh
#
# Builds the FMT-profile-driven background metagenome pool using
# InSilicoSeq (ISS), replacing the archived generate_background.sh's
# built-in lognormal/exponential abundance distributions with real
# per-genome sequencing depth derived from an actual FMT sample's Bracken-
# corrected kraken2 classification (config.OUT_FMT_ABUNDANCE_PROFILES,
# scripts/build_fmt_abundance_profile.py). Draft genomes are PHORAGER's
# prophage-stripped HumGut sequences (config.HUMGUT_PROPHAGE_REMOVED_DIR,
# scripts/extract_humgut_phoraged.py), not raw HumGut MAGs.
#
# Grid: config.FMT_BACKGROUND_MODELS x FMT_READS_SAMPLE_LIST x
# FMT_BACKGROUND_N_READS x FMT_BACKGROUND_REPLICATES (1 x 3 x 1 x 3 = 9
# combos), one SLURM array task per combo. Replicates are independent ISS
# runs of the SAME (model, sample, n_reads) combo -- same target abundance
# profile, different random seed -- added 2026-09-22 so simulation noise can
# be separated from real sample-to-sample biological variation. --draft file
# list is derived directly from each sample's own abundance profile's first
# column (`cut -f1`) -- NOT reconstructed independently -- since the
# profile's genome_id already IS the real resolved path ISS needs, and
# rebuilding it separately risks silently drifting out of sync with what's
# actually in the profile.
#
# Read length is never hardcoded: after each run this script measures the
# actual R1 read length and read-pair count from the generated FASTQ and
# appends a row to config.BACKGROUND_FMT_MANIFEST, in the exact schema
# read_mixing.sh already expects (combo_name, model, abundance, n_reads,
# pairs, read_len, seed) -- "abundance" holds the sample ID that drove the
# profile here instead of a distribution name; read_mixing.sh never uses
# that field for anything beyond destructuring the row, so this is safe.
#
# Idempotent: skips a combo if its manifest row + output files already exist.
#
# Usage:
#   sbatch --array=0-8 scripts/generate_background_fmt.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=generate_background_fmt
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

set -euo pipefail

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
ISS_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/insilicoseq/bin/iss"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"

# Loaded from config.py (single source of truth) rather than hardcoded here.
PROFILES_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.OUT_FMT_ABUNDANCE_PROFILES)")
BACKGROUND_DIR=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.OUT_BACKGROUND_FMT)")
MANIFEST=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.BACKGROUND_FMT_MANIFEST)")
SAMPLE_LIST=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_READS_SAMPLE_LIST)")

mkdir -p "$PROJECT/logs/slurm" "$BACKGROUND_DIR"

(
    flock -x 200
    if [ ! -s "$MANIFEST" ]; then
        printf "combo_name\tmodel\tabundance\tn_reads_requested\tn_pairs_actual\tread_length_actual\tseed\n" > "$MANIFEST"
    fi
) 200>"$MANIFEST.lock"

# --- Validate prerequisites ---------------------------------------------------

if [ ! -d "$PROFILES_DIR" ] || [ -z "$(ls -A "$PROFILES_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $PROFILES_DIR is missing or empty."
    echo "Run scripts/build_fmt_abundance_profile.py first."
    exit 1
fi

mapfile -t SAMPLES < "$SAMPLE_LIST"
if [ "${#SAMPLES[@]}" -eq 0 ]; then
    echo "ERROR: $SAMPLE_LIST is empty."
    exit 1
fi

mapfile -t MODELS < <(
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT')
import config
for m in config.FMT_BACKGROUND_MODELS:
    print(m)
"
)
mapfile -t N_READS_LIST < <(
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT')
import config
for n in config.FMT_BACKGROUND_N_READS:
    print(n)
"
)
REPLICATES=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.FMT_BACKGROUND_REPLICATES)")

# --- Build the model x sample x n_reads x replicate grid ----------------------

combos=()
for model in "${MODELS[@]}"; do
    for sample in "${SAMPLES[@]}"; do
        for n_reads in "${N_READS_LIST[@]}"; do
            for replicate in $(seq 1 "$REPLICATES"); do
                combos+=("${model}:${sample}:${n_reads}:${replicate}")
            done
        done
    done
done
expected_combos=$(( ${#MODELS[@]} * ${#SAMPLES[@]} * ${#N_READS_LIST[@]} * REPLICATES ))
echo "Total combos: ${#combos[@]} (expect $expected_combos, from ${#MODELS[@]} models x ${#SAMPLES[@]} samples x ${#N_READS_LIST[@]} read depths x $REPLICATES replicates)"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: not running as a SLURM array job. Submit with:"
    echo "  sbatch --array=0-$(( ${#combos[@]} - 1 )) $0"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#combos[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of combos (${#combos[@]}). Nothing to do."
    exit 0
fi

IFS=':' read -r model sample n_reads replicate <<< "${combos[$SLURM_ARRAY_TASK_ID]}"
combo_name="${sample}_${model}_${n_reads}_rep${replicate}"
out_dir="$BACKGROUND_DIR/$combo_name"
out_prefix="$out_dir/background"

echo "Combo: model=$model sample=$sample n_reads=$n_reads replicate=$replicate"

profile_tsv="$PROFILES_DIR/${sample}_abundance.tsv"
if [ ! -f "$profile_tsv" ]; then
    echo "ERROR: no abundance profile for $sample at $profile_tsv"
    echo "Run scripts/build_fmt_abundance_profile.py first."
    exit 1
fi

# --- Skip if already complete -------------------------------------------------

if [ -f "${out_prefix}_R1.fastq.gz" ] && [ -f "${out_prefix}_R2.fastq.gz" ] \
   && grep -qP "^${combo_name}\t" "$MANIFEST" 2>/dev/null; then
    echo "Skipping $combo_name -- already complete."
    exit 0
fi

mkdir -p "$out_dir"

# --draft file list comes directly from the profile's own first column --
# these are already the real, resolved paths to each genome's extracted
# phoraged FASTA (see scripts/build_fmt_abundance_profile.py).
mapfile -t DRAFT_FILES < <(cut -f1 "$profile_tsv")
if [ "${#DRAFT_FILES[@]}" -eq 0 ]; then
    echo "ERROR: no genomes listed in $profile_tsv"
    exit 1
fi
echo "Draft genomes for $sample: ${#DRAFT_FILES[@]}"

seed=$(( 1000 + SLURM_ARRAY_TASK_ID ))

# --- Run ISS -------------------------------------------------------------------

"$ISS_BIN" generate \
    --draft "${DRAFT_FILES[@]}" \
    --model "$model" \
    --abundance_file "$profile_tsv" \
    --n_reads "$n_reads" \
    --seed "$seed" \
    --cpus "${SLURM_CPUS_PER_TASK:-8}" \
    --compress \
    --output "$out_prefix"

if [ ! -f "${out_prefix}_R1.fastq.gz" ]; then
    echo "ERROR: ISS failed for $combo_name -- no output R1 produced."
    exit 1
fi

# --- Measure actual read length + pair count ----------------------------------
# awk's early `exit` closes the pipe before zcat finishes writing, so zcat
# gets SIGPIPE (exit 141) -- harmless normally, but fatal under `set -o
# pipefail` (which the archived generate_background.sh this was based on
# never set). Toggle pipefail off just for this one command.

set +o pipefail
read_length=$(zcat "${out_prefix}_R1.fastq.gz" | awk 'NR==2{print length($0); exit}')
set -o pipefail
n_pairs_actual=$(zcat "${out_prefix}_R1.fastq.gz" | awk 'END{print NR/4}')

if [ -z "$read_length" ] || [ "$read_length" -le 0 ]; then
    echo "ERROR: could not measure read length for $combo_name"
    exit 1
fi

echo "Measured: read_length=$read_length n_pairs_actual=$n_pairs_actual"

# --- Append to manifest (flock-guarded: 9 array tasks write concurrently) -----

(
    flock -x 200
    if ! grep -qP "^${combo_name}\t" "$MANIFEST" 2>/dev/null; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$combo_name" "$model" "$sample" "$n_reads" \
            "$n_pairs_actual" "$read_length" "$seed" >> "$MANIFEST"
    fi
) 200>"$MANIFEST.lock"

echo "Finished $combo_name at $(date)"
