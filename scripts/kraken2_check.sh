#!/usr/bin/env bash
# =============================================================================
# kraken2_check.sh
#
# Gate for the spike-in benchmark, run after read-mixing but before assembly:
# confirms every viral reference actually spiked into the generated samples is
# detectable by kraken2, using the REAL post-mix reads (not a synthetic
# simulation) -- before the (much heavier) assembly/viral-detection pipeline
# runs on top of them. Deliberately agnostic to whichever viral reference
# catalog was used to build the spike-in samples (AVrC, INPHARED, or
# otherwise) -- it works off whatever refs_log.txt pick_refs.py already wrote
# in each sample dir, with no hardcoded dependency on a specific catalog's
# FASTA or ID scheme.
#
# NOTE on approach: reference catalogs can use custom IDs that don't overlap
# NCBI accessions at all (verified 2026-07-07 for AVrC's catalog IDs, e.g.
# "GutCatV1_GPD_113896", against kraken2_pluspfp's seqid2taxid.map, which is
# keyed on NCBI RefSeq accessions like NC_001422.1). An ID-presence lookup is
# therefore not a reliable general check; the only real signal is whether
# kraken2 can classify real reads from each genome at all. This script:
#   1. Pools every sample/pct combo's final.1.fq.gz as-is (no
#      pre-filtering, no re-simulation -- these are the exact reads
#      read_mixing.sh already produced and combine_reads.py already mixed
#      with background) and streams them straight into kraken2.
#   2. Runs ONE kraken2 classify call over the whole pool (loading the
#      ~16GB pluspfp hash once, not once per sample/combo).
#   3. Attributes each classified/unclassified read back to its source genome
#      by parsing kraken2's own read-ID column. Viral spike-in reads are
#      ART-simulated (scripts/read_mixing.sh) with a read-naming convention
#      verified empirically 2026-07-15 (live `art_illumina` test, not just
#      docs): "<genome_id>-<read_number>/1" (mate 2 ends "/2") -- e.g. a
#      genome header ">GutCatV1_GPD_113896" produces read IDs like
#      "GutCatV1_GPD_113896-20/1". This survives combine_reads.py's shuffle
#      untouched since it never rewrites headers, and survives kraken2
#      classify since its --output echoes the read ID verbatim. Background
#      reads keep ISS's own convention ("<genome_id>_<counter>_<chunk>/1"),
#      but that's irrelevant here -- background genome IDs never match a
#      known viral reference, so they're ignored regardless of how they
#      parse. Reads whose derived genome ID isn't one of the sample's known
#      viral references (i.e. background reads) are ignored.
#   4. Reports per-reference classified-read recall; exits nonzero if any
#      reference has zero classified reads, so submit_pipeline.sh's
#      dependency chain blocks assembly on a hard failure here.
#
# Usage:
#   sbatch scripts/kraken2_check.sh
# =============================================================================
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

echo "Job started at $(date)"
echo "Running on node $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
SPIKE_IN_DIR="$PROJECT/data/spike_in_samples"
KRAKEN2_SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_kraken2:2.17.1--pl5321h077b44d_0.sif"
KRAKEN2_DB="/rs1/shares/brc/admin/databases/kraken2_pluspfp"
OUT_DIR="$PROJECT/results/00_kraken2_check"

mkdir -p "$PROJECT/logs/slurm" "$OUT_DIR"
module load apptainer/1.4.2-1

# --- Validate prerequisites ---------------------------------------------------

if [ ! -f "$KRAKEN2_SIF" ]; then
    echo "ERROR: kraken2 container not found at $KRAKEN2_SIF"
    exit 1
fi
if [ ! -d "$KRAKEN2_DB" ]; then
    echo "ERROR: kraken2 DB not found at $KRAKEN2_DB"
    exit 1
fi

mapfile -t SAMPLE_DIRS < <(find "$SPIKE_IN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'sample_*v_*' | sort)
if [ "${#SAMPLE_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: no sample_<size>v_<n> directories found in $SPIKE_IN_DIR."
    echo "Run scripts/pick_refs.py first."
    exit 1
fi

mapfile -t REF_IDS < <(cat "$SPIKE_IN_DIR"/sample_*v_*/refs_log.txt 2>/dev/null | sort -u)
if [ "${#REF_IDS[@]}" -eq 0 ]; then
    echo "ERROR: no refs_log.txt found under $SPIKE_IN_DIR/sample_*v_*/."
    echo "Run scripts/pick_refs.py first."
    exit 1
fi
echo "Checking ${#REF_IDS[@]} unique viral references."

TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-kraken2check}"
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

# --- Find every combo's final.1.fq.gz across every sample ------------------
# No filtering here -- kraken2 classifies the full real mixed read set
# (background included); attribution back to specific viral references
# happens afterward by parsing kraken2's own output.

printf '%s\n' "${REF_IDS[@]}" > "$TMPDIR/ref_ids.txt"

ALL_FINAL_FILES=()
for sample_dir in "${SAMPLE_DIRS[@]}"; do
    sample_name=$(basename "$sample_dir")

    mapfile -t FINAL_FILES < <(find "$sample_dir" -name 'final.1.fq.gz' | sort)
    if [ "${#FINAL_FILES[@]}" -eq 0 ]; then
        echo "WARNING: no final.1.fq.gz found under $sample_dir yet -- skipping"
        continue
    fi

    ALL_FINAL_FILES+=("${FINAL_FILES[@]}")
    echo "Found $sample_name (${#FINAL_FILES[@]} combos)."
done

if [ "${#ALL_FINAL_FILES[@]}" -eq 0 ]; then
    echo "ERROR: no final.1.fq.gz files found anywhere under $SPIKE_IN_DIR."
    echo "Run scripts/read_mixing.sh first."
    exit 1
fi
echo "Classifying ${#ALL_FINAL_FILES[@]} FASTQ files (background + viral reads, real post-mix data)."

# --- One kraken2 classify call over everything (load the DB once) -----------
# Streamed straight in via a pipe rather than materialized as one combined
# FASTQ on disk first -- this pool spans every sample/pct combo's full
# real read set (background included), which can be large.

zcat "${ALL_FINAL_FILES[@]}" | apptainer exec --bind /rs1 "$KRAKEN2_SIF" kraken2 \
    --db "$KRAKEN2_DB" --threads "$SLURM_CPUS_PER_TASK" \
    --output "$OUT_DIR/kraken2_out.txt" --report "$OUT_DIR/kraken2_report.txt" \
    /dev/stdin

if [ ! -f "$OUT_DIR/kraken2_out.txt" ]; then
    echo "ERROR: kraken2 classify failed."
    exit 1
fi

# --- Attribute per-reference recall by parsing kraken2's own read IDs -------
# Viral reads are ART-simulated ("<genome_id>-<read_number>/1"); kraken2's
# --output echoes the read ID verbatim. Reads whose derived genome ID isn't
# one of our known viral references (i.e. background reads, still in ISS's
# own convention) are ignored regardless of how they parse.

"$PYTHON_BIN" - "$OUT_DIR/kraken2_out.txt" "$TMPDIR/ref_ids.txt" "$OUT_DIR/per_reference_recall.tsv" <<'PYEOF'
import re
import sys
from collections import defaultdict
from pathlib import Path

out_path, ids_path, report_path = sys.argv[1:4]

with open(ids_path) as f:
    ref_ids = {line.strip() for line in f if line.strip()}

MATE_RE = re.compile(r"/[12]$")
ART_COUNTER_RE = re.compile(r"-[0-9]+$")  # ART's own "<genome_id>-<read_number>"


def genome_of(read_id):
    id_no_mate = MATE_RE.sub("", read_id)
    return ART_COUNTER_RE.sub("", id_no_mate)


classified = defaultdict(int)
total = defaultdict(int)

with open(out_path) as f:
    for line in f:
        fields = line.rstrip("\n").split("\t")
        status, read_id = fields[0], fields[1]
        genome = genome_of(read_id)
        if genome not in ref_ids:
            continue
        total[genome] += 1
        if status == "C":
            classified[genome] += 1

ref_ids = sorted(ref_ids)

no_reads_generated = []
zero_classified = []
with open(report_path, "w") as out_f:
    out_f.write("reference_id\ttotal_reads\tclassified_reads\trecall\n")
    for ref_id in ref_ids:
        t = total.get(ref_id, 0)
        c = classified.get(ref_id, 0)
        recall = (c / t) if t else 0.0
        out_f.write(f"{ref_id}\t{t}\t{c}\t{recall:.4f}\n")
        if t == 0:
            no_reads_generated.append(ref_id)
        elif c == 0:
            zero_classified.append(ref_id)

print(f"Wrote per-reference recall to {report_path}")
if no_reads_generated:
    print(
        f"{len(no_reads_generated)} reference(s) got ZERO real spike-in reads across "
        "every sample/pct combo they appear in (a data-generation issue, not "
        "necessarily a kraken2 detectability issue):"
    )
    for r in no_reads_generated:
        print(f"  {r}")
if zero_classified:
    print(f"{len(zero_classified)} reference(s) had real reads but ZERO classified by kraken2:")
    for r in zero_classified:
        print(f"  {r}")
if no_reads_generated or zero_classified:
    sys.exit(1)
PYEOF
status=$?

echo "Finished at $(date)"
exit $status
