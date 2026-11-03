#!/usr/bin/env bash
# Simulate one background metagenome per (model, sample, n_reads, replicate) with InSilicoSeq,
# from each sample's abundance profile (06), and record it in the background manifest.
# Usage: sbatch --array=0-$((N-1)) mock_datasets/scripts/07_generate_background_fmt.sh   # N = number of combos (3 now)
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

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
ISS_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/insilicoseq/bin/iss"
cfg() { "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.$1)"; }

PROFILES_DIR=$(cfg OUT_FMT_ABUNDANCE_PROFILES)
BACKGROUND_DIR=$(cfg OUT_BACKGROUND_FMT)
MANIFEST=$(cfg BACKGROUND_FMT_MANIFEST)

# model sample n_reads replicate, one line per array task
mapfile -t COMBOS < <("$PYTHON_BIN" -c "
import itertools, sys; sys.path.insert(0, '$PROJECT'); import config
samples = config.FMT_READS_SAMPLE_LIST.read_text().split()
reps = range(1, config.FMT_BACKGROUND_REPLICATES + 1)
for combo in itertools.product(config.FMT_BACKGROUND_MODELS, samples, config.FMT_BACKGROUND_N_READS, reps):
    print(*combo)
")
read -r model sample n_reads replicate <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
name="${sample}_${model}_${n_reads}_rep${replicate}"
out="$BACKGROUND_DIR/$name/background"
seed=$(( 1000 + SLURM_ARRAY_TASK_ID ))
profile="$PROFILES_DIR/${sample}_abundance.tsv"
mkdir -p "$BACKGROUND_DIR/$name"

grep -q "^${name}"$'\t' "$MANIFEST" 2>/dev/null && { echo "$name done, skipping"; exit 0; }

# ISS matches the profile's first column against the literal --draft paths, so --draft comes from that column
"$ISS_BIN" generate \
    --draft $(cut -f1 "$profile") \
    --abundance_file "$profile" \
    --model "$model" \
    --n_reads "$n_reads" \
    --seed "$seed" \
    --cpus "${SLURM_CPUS_PER_TASK:-8}" \
    --compress \
    --output "$out"

read_len=$( (zcat "${out}_R1.fastq.gz" || true) | awk 'NR == 2 { print length($0); exit }')
n_pairs=$(zcat "${out}_R1.fastq.gz" | awk 'END { print NR / 4 }')

(
    flock -x 200
    [ -s "$MANIFEST" ] || printf "combo_name\tmodel\tabundance\tn_reads_requested\tn_pairs_actual\tread_length_actual\tseed\n" > "$MANIFEST"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$model" "$sample" "$n_reads" "$n_pairs" "$read_len" "$seed" >> "$MANIFEST"
) 200>"$MANIFEST.lock"
