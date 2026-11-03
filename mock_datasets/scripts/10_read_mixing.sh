#!/usr/bin/env bash
# Build every condition in one sample dir's conditions.tsv (from 09): ART Illumina reads per genome at
# that condition's coverages, shuffled into a random usable background (07) as {condition}/final.{1,2}.fq.gz.
# Usage: N=$(find data/spike_in_samples -name conditions.tsv | wc -l)
#        sbatch --array=0-$((N-1)) mock_datasets/scripts/10_read_mixing.sh
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

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
ART_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/art_illumina"
cfg() { "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.$1)"; }

SPIKE_IN_DIR=$(cfg SPIKE_IN_DIR)
BACKGROUND_DIR=$(cfg OUT_BACKGROUND_FMT)
FRAGMENT_MEAN=$(cfg ART_FRAGMENT_MEAN)
FRAGMENT_SD=$(cfg ART_FRAGMENT_SD)

# Background rows whose read length the model's ART profile supports: combo, ART profile, pairs, read length
mapfile -t BACKGROUNDS < <("$PYTHON_BIN" -c "
import csv, sys; sys.path.insert(0, '$PROJECT'); import config
for row in csv.DictReader(open(config.BACKGROUND_FMT_MANIFEST), delimiter='\t'):
    profile = config.ART_PROFILE_BY_MODEL.get(row['model'])
    if profile and int(row['read_length_actual']) <= config.ART_PROFILE_MAX_LEN[profile]:
        print(row['combo_name'], profile, row['n_pairs_actual'], row['read_length_actual'], sep='\t')
")

mapfile -t SAMPLE_DIRS < <(find "$SPIKE_IN_DIR" -name conditions.tsv -printf '%h\n' | sort)
sample_dir="${SAMPLE_DIRS[$SLURM_ARRAY_TASK_ID]}"

TMP="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

# Same command as nextflow/art_illumina.nf: every sequence in a FASTA at one fold coverage -> <sampleid>_1.fq, _2.fq
# (-ss and -l come from the chosen background here, from params in the process)
art_illumina() {
    local sampleid=$1 fasta=$2 fold_coverage=$3
    "$ART_BIN" \
        -ss "$profile" \
        -p \
        -na \
        -i "$fasta" \
        -f "$fold_coverage" \
        -l "$read_len" \
        -m "$FRAGMENT_MEAN" \
        -s "$FRAGMENT_SD" \
        -o "${sampleid}_"
}

# Workflow bookkeeping, done here in bash:
# 1. split contigs.fasta into one FASTA per genome (Nextflow: splitFasta)
ids=( $(grep '^>' "$sample_dir/contigs.fasta" | sed 's/^>//; s/[[:space:]].*//') )
awk '/^>/ { if (out) close(out); out = substr($1, 2) ".fa" } { print > out }' "$sample_dir/contigs.fasta"

while IFS=$'\t' read -r -u 3 cond coverages; do
    out="$sample_dir/$cond"
    [ -f "$out/final.2.fq.gz" ] && { echo "$cond done, skipping"; continue; }
    mkdir -p "$out"

    IFS=$'\t' read -r bg profile bg_pairs read_len <<< "${BACKGROUNDS[$(( RANDOM % ${#BACKGROUNDS[@]} ))]}"
    IFS=',' read -ra covs <<< "$coverages"
    rm -f -- *.fq

    # 2. pair each genome with its coverage and 3. run ART once per genome (Nextflow: one art_illumina task each)
    printf "genome_id\tn_pairs_actual\tgenome_length\tcoverage_x\n" > "$out/coverage_log.txt"
    for i in "${!ids[@]}"; do
        sampleid="${cond}_${ids[$i]}"
        art_illumina "$sampleid" "${ids[$i]}.fa" "${covs[$i]}"

        # Coverage actually achieved, from ART's read count (QC only)
        pairs=$(( $(wc -l < "${sampleid}_1.fq") / 4 ))
        length=$(grep -v '^>' "${ids[$i]}.fa" | tr -d '\n' | wc -c)
        awk -v id="${ids[$i]}" -v p="$pairs" -v l="$length" -v r="$read_len" \
            'BEGIN { printf "%s\t%d\t%d\t%.4f\n", id, p, l, p * 2 * r / l }' >> "$out/coverage_log.txt"
    done

    # 4. join this condition's per-genome reads into one pair (Nextflow: collectFile)
    cat "${cond}"_*_1.fq > viral.1.fq
    cat "${cond}"_*_2.fq > viral.2.fq
    viral_pairs=$(( $(wc -l < viral.1.fq) / 4 ))
    awk -v c="$cond" -v cov="$coverages" -v bg="$bg" -v v="$viral_pairs" -v b="$bg_pairs" 'BEGIN {
        print "condition\tcoverages\tbg_combo\tviral_pairs\tbg_pairs\tviral_pct"
        printf "%s\t%s\t%s\t%d\t%d\t%.4f\n", c, cov, bg, v, b, 100 * v / (v + b)
    }' > "$out/spike_summary.tsv"

    # Shuffle viral + background read pairs together (one line per pair: R1 and R2 records side by side)
    paste <(cat viral.1.fq; zcat "$BACKGROUND_DIR/$bg/background_R1.fastq.gz") \
          <(cat viral.2.fq; zcat "$BACKGROUND_DIR/$bg/background_R2.fastq.gz") \
        | paste - - - - \
        | shuf \
        | awk -F'\t' -v OFS='\n' '{ print $1, $3, $5, $7 > "final.1.fq"; print $2, $4, $6, $8 > "final.2.fq" }'
    gzip -f final.1.fq final.2.fq
    mv final.1.fq.gz final.2.fq.gz "$out/"
done 3< "$sample_dir/conditions.tsv"
