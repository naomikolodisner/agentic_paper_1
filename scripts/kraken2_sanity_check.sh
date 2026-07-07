#!/usr/bin/env bash
# =============================================================================
# kraken2_sanity_check.sh
#
# First-step gate for the spike-in benchmark: confirms every viral reference
# used across the generated spike-in samples is actually detectable by
# kraken2, before the (much heavier) read-mixing/assembly/viral-detection
# pipeline runs on top of them.
#
# NOTE on approach: AVrC references use custom viral-catalog IDs (e.g.
# "GutCatV1_GPD_113896"), not NCBI accessions -- verified 2026-07-07 that none
# of these IDs appear in kraken2_pluspfp's seqid2taxid.map (which is keyed on
# NCBI RefSeq accessions like NC_001422.1). An ID-presence lookup is therefore
# meaningless here; the only real signal is whether kraken2 can classify
# simulated reads from each genome at all. This script:
#   1. Dedupes viral reference IDs across every generated sample's refs_log.txt.
#   2. Simulates a small, fixed-size ISS read set per reference (one model,
#      one seed per ref for reproducibility), tagging each read's header with
#      its source genome ID.
#   3. Runs ONE kraken2 classify call over all references' reads combined
#      (loading the ~16GB pluspfp hash once, not once per reference).
#   4. Reports per-reference classified-read recall; exits nonzero if any
#      reference has zero classified reads, so submit_pipeline.sh's
#      dependency chain blocks read-mixing on a hard failure here.
#
# Usage:
#   sbatch scripts/kraken2_sanity_check.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=kraken2_sanity_check
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

echo "Job started at $(date)"
echo "Running on node $(hostname)"

# --- Paths -------------------------------------------------------------------

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
ISS_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/insilicoseq/bin/iss"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
SPIKE_IN_DIR="$PROJECT/data/spike_in_samples"
AVRC_ALL_SEQUENCES="/rs1/researchers/b/blhurwit/users/nkolodi/databases/AVrC/AVrC_allsequences.fasta"
KRAKEN2_SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_kraken2:2.17.1--pl5321h077b44d_0.sif"
KRAKEN2_DB="/rs1/shares/brc/admin/databases/kraken2_pluspfp"
OUT_DIR="$PROJECT/results/00_kraken2_sanity_check"
N_READS_PER_REF=2000
MODEL=hiseq

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

# --- Extract each reference's sequence into its own single-record FASTA ------

printf '%s\n' "${REF_IDS[@]}" > "$TMPDIR/ref_ids.txt"
"$PYTHON_BIN" - "$AVRC_ALL_SEQUENCES" "$TMPDIR/ref_ids.txt" "$TMPDIR/refs" <<'PYEOF'
import sys
from pathlib import Path
from Bio import SeqIO

fasta_path, ids_path, out_dir = sys.argv[1:4]
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

with open(ids_path) as f:
    ref_ids = [line.strip() for line in f if line.strip()]

index = SeqIO.index(fasta_path, "fasta")
missing = []
for ref_id in ref_ids:
    if ref_id not in index:
        missing.append(ref_id)
        continue
    with open(out_dir / f"{ref_id}.fasta", "w") as out_f:
        SeqIO.write(index[ref_id], out_f, "fasta")

if missing:
    print(f"WARNING: {len(missing)} reference(s) not found in {fasta_path}:")
    for m in missing:
        print(f"  {m}")
    Path(out_dir / "MISSING.txt").write_text("\n".join(missing) + "\n")
PYEOF

# --- Simulate a small fixed-size read set per reference, tag headers with the genome ID ---

COMBINED_FASTQ="$TMPDIR/combined.fastq"
> "$COMBINED_FASTQ"

seed=42
for ref_fasta in "$TMPDIR"/refs/*.fasta; do
    [ -f "$ref_fasta" ] || continue
    ref_id=$(basename "$ref_fasta" .fasta)
    seed=$((seed + 1))

    "$ISS_BIN" generate --genomes "$ref_fasta" --model "$MODEL" \
        --n_reads "$N_READS_PER_REF" --seed "$seed" --cpus "$SLURM_CPUS_PER_TASK" \
        --output "$TMPDIR/one_ref" >/dev/null 2>&1

    if [ ! -f "$TMPDIR/one_ref_R1.fastq" ]; then
        echo "WARNING: ISS failed to simulate reads for $ref_id -- skipping"
        continue
    fi

    # Tag every read header with its source genome ID so we can attribute
    # kraken2's per-read classification back to the reference afterward.
    awk -v ref="$ref_id" 'NR%4==1{sub(/^@/, "@" ref "::"); print; next} {print}' \
        "$TMPDIR/one_ref_R1.fastq" >> "$COMBINED_FASTQ"

    rm -f "$TMPDIR/one_ref"*
done

n_combined=$(awk 'END{print NR/4}' "$COMBINED_FASTQ")
echo "Simulated $n_combined total reads across ${#REF_IDS[@]} references."

# --- One kraken2 classify call over everything (load the DB once) -----------

apptainer exec --bind /rs1 "$KRAKEN2_SIF" kraken2 \
    --db "$KRAKEN2_DB" --threads "$SLURM_CPUS_PER_TASK" \
    --output "$OUT_DIR/kraken2_out.txt" --report "$OUT_DIR/kraken2_report.txt" \
    "$COMBINED_FASTQ"

if [ ! -f "$OUT_DIR/kraken2_out.txt" ]; then
    echo "ERROR: kraken2 classify failed."
    exit 1
fi

# --- Attribute per-reference recall from the tagged read IDs -----------------

"$PYTHON_BIN" - "$OUT_DIR/kraken2_out.txt" "$TMPDIR/ref_ids.txt" "$OUT_DIR/per_reference_recall.tsv" <<'PYEOF'
import sys
from collections import defaultdict
from pathlib import Path

out_path, ids_path, report_path = sys.argv[1:4]

with open(ids_path) as f:
    ref_ids = [line.strip() for line in f if line.strip()]

classified = defaultdict(int)
total = defaultdict(int)

with open(out_path) as f:
    for line in f:
        fields = line.rstrip("\n").split("\t")
        status, read_id = fields[0], fields[1]
        ref_id = read_id.split("::", 1)[0]
        total[ref_id] += 1
        if status == "C":
            classified[ref_id] += 1

zero_detected = []
with open(report_path, "w") as out_f:
    out_f.write("reference_id\ttotal_reads\tclassified_reads\trecall\n")
    for ref_id in ref_ids:
        t = total.get(ref_id, 0)
        c = classified.get(ref_id, 0)
        recall = (c / t) if t else 0.0
        out_f.write(f"{ref_id}\t{t}\t{c}\t{recall:.4f}\n")
        if t > 0 and c == 0:
            zero_detected.append(ref_id)
        elif t == 0:
            zero_detected.append(ref_id)

print(f"Wrote per-reference recall to {report_path}")
if zero_detected:
    print(f"{len(zero_detected)} reference(s) with ZERO classified reads:")
    for r in zero_detected:
        print(f"  {r}")
    sys.exit(1)
PYEOF
status=$?

echo "Finished at $(date)"
exit $status
