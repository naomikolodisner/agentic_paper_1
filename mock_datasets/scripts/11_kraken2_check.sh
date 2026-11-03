#!/usr/bin/env bash
# Gate before assembly: classify every spike-in sample's reads with kraken2 in one pooled call, then count
# classified reads per spiked reference (ART read IDs are "<genome_id>-<n>/1"; background reads are ignored).
# Fails if any reference got no reads or had none classified.
# Usage: sbatch mock_datasets/scripts/11_kraken2_check.sh
#SBATCH --partition=compute
#SBATCH --job-name=kraken2_check
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nkolodi@ncsu.edu

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_kraken2:2.17.1--pl5321h077b44d_0.sif"
cfg() { "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.$1)"; }

SPIKE_IN_DIR=$(cfg SPIKE_IN_DIR)
KRAKEN2_DB=$(cfg KRAKEN2_DB)
OUT="$PROJECT/mock_datasets/results/kraken2_check"
mkdir -p "$OUT"
module load apptainer/1.4.2-1

find "$SPIKE_IN_DIR" -name refs_log.txt -exec cat {} + | sort -u > "$OUT/ref_ids.txt"

find "$SPIKE_IN_DIR" -name final.1.fq.gz -print0 | xargs -0 -r zcat \
    | apptainer exec --bind /rs1 "$SIF" kraken2 \
        --db "$KRAKEN2_DB" \
        --threads "${SLURM_CPUS_PER_TASK:-16}" \
        --output "$OUT/kraken2_out.txt" \
        --report "$OUT/kraken2_report.txt" \
        /dev/stdin

awk -F'\t' '
    NR == FNR { ref[$1]; next }
    { id = $2; sub(/\/[12]$/, "", id); sub(/-[0-9]+$/, "", id) }
    id in ref { total[id]++; if ($1 == "C") classified[id]++ }
    END {
        for (r in ref) {
            t = total[r] + 0; c = classified[r] + 0
            printf "%s\t%d\t%d\t%.4f\n", r, t, c, (t ? c / t : 0)
            if (t == 0) { print "No spike-in reads: " r > "/dev/stderr"; bad = 1 }
            else if (c == 0) { print "No reads classified: " r > "/dev/stderr"; bad = 1 }
        }
        exit bad
    }' "$OUT/ref_ids.txt" "$OUT/kraken2_out.txt" \
    | sort | cat <(printf "reference_id\ttotal_reads\tclassified_reads\trecall\n") - \
    > "$OUT/per_reference_recall.tsv"
