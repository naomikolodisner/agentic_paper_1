#!/usr/bin/env bash
# =============================================================================
# bracken_build.sh
#
# One-time build of the Bracken kmer-distribution index for
# config.HUMGUT_KRAKEN2_DB (data/humgut_kraken2/), at read length 100 (our
# real FMT reads are 101bp). This lets scripts/kraken2_fmt_reads.sh's
# per-sample .report files be corrected with `bracken`: ~32% of classified
# reads in a real test sample (ERR2198703) landed at an ancestor rank
# (genus/family/order/...) rather than on a specific HumGut genome, because
# closely related genomes in HumGut share k-mers and kraken2 has to call
# those reads at their lowest common ancestor. Bracken redistributes those
# ancestor-level reads back down onto the genomes actually detected in a
# sample, using a probability table built here (once, from the DB itself --
# not from any sample's data).
#
# PREREQUISITE SETUP (this DB's on-disk layout doesn't match what
# bracken-build expects -- verified by reading the vendored script, not
# assumed): bracken-build requires $DB/library/ (raw genome FASTA,
# uncompressed, .fna/.fa/.fasta), $DB/taxonomy/{nodes,names}.dmp, and
# $DB/seqid2taxid.map. None of these exist at data/humgut_kraken2/ right now
# -- the taxonomy dumps sit at the DB's top level as GTDB_names.dmp /
# GTDB_nodes.dmp (not under taxonomy/), and the original concatenated
# library FASTA (HumGut975_library.fna) was deliberately deleted after the
# original kraken2-build --build to save 64GB (data/humgut_kraken2/fna/ only
# has the individual per-genome .fna.gz originals). This script rebuilds
# both:
#   - taxonomy/{nodes,names}.dmp -> symlinked to the existing GTDB_*.dmp
#   - library/library.fna -> re-decompressed from fna/*.fna.gz, concatenated
#     into ONE file (not 31225 separate files) specifically to avoid
#     ARG_MAX: bracken-build passes every file under library/ as a separate
#     argv entry to kraken2, and 31225 file paths at this DB's path length
#     comes to ~3.4MB of argv, over this system's 2MB ARG_MAX (verified via
#     `getconf ARG_MAX` before writing this script) -- a single concatenated
#     file sidesteps that entirely (kraken2 classifies by FASTA record, not
#     by file, so this changes nothing about the actual classification).
#     Decompression is parallelized across SLURM_CPUS_PER_TASK by splitting
#     the file list into per-worker chunks (each worker appends to its own
#     temp part-file, safe under concurrent writes), then concatenating the
#     parts in order.
#   - seqid2taxid.map -> parsed directly from the decompressed library's own
#     ">kraken:taxid|<taxid>|<seqid> ..." headers (this DB's build
#     convention, confirmed by inspecting several genome files directly).
#     KEY MUST BE THE FULL HEADER TOKEN ("kraken:taxid|<taxid>|<seqid>"),
#     NOT just the stripped <seqid> -- verified by reading
#     data/humgut_kraken2/database.kraken directly: kraken2 classify keeps
#     the whole first header token (including the "kraken:taxid|N|" prefix)
#     as the sequence ID; that stripping only happens at kraken2-build
#     --add-to-library time, not at classify time. An earlier version of
#     this script (job 888905) keyed on the stripped seqid instead, which
#     made every kmer2read_distr --seqid2taxid lookup fail silently and
#     fall back to one constant bogus "genome" (273997528, not even in our
#     4xxxxxx custom taxid range) for every sequence -- confirmed by
#     inspecting database100mers.kraken/.kmer_distrib directly, both showing
#     that one fake value on every line. Running `bracken` against that
#     corrupted table produced "Reads distributed: 0" for all 3 test
#     samples. Fixed and re-verified 2026-09-20 (formerly a separate
#     scripts/bracken_build_fix.sh patch, folded back in here -- see
#     scripts/archive/bracken_build_fix.sh for that history).
#
# Output (written into data/humgut_kraken2/ by bracken-build itself):
#   database.kraken, database100mers.kraken, database100mers.kmer_distrib
#
# Usage:
#   sbatch scripts/bracken_build.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=bracken_build
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=nkolodi@ncsu.edu

echo "Job started at $(date) on $(hostname)"
set -euo pipefail

# --- Paths -------------------------------------------------------------------

KRAKEN2_DB="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/humgut_kraken2"
FNA_DIR="$KRAKEN2_DB/fna"
SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_bracken:3.1--h9948957_0.sif"
NPROC="${SLURM_CPUS_PER_TASK:-10}"

module load apptainer/1.4.2-1

if [ ! -f "$SIF" ]; then
    echo "ERROR: bracken container not found at $SIF"
    exit 1
fi
if [ ! -f "$KRAKEN2_DB/hash.k2d" ]; then
    echo "ERROR: kraken2 DB not found/incomplete at $KRAKEN2_DB (missing hash.k2d)"
    exit 1
fi

# --- Rebuild taxonomy/ (symlinks -- no copy needed) ---------------------------

mkdir -p "$KRAKEN2_DB/taxonomy" "$KRAKEN2_DB/library"
ln -sf "$KRAKEN2_DB/GTDB_nodes.dmp" "$KRAKEN2_DB/taxonomy/nodes.dmp"
ln -sf "$KRAKEN2_DB/GTDB_names.dmp" "$KRAKEN2_DB/taxonomy/names.dmp"

# --- Rebuild library/library.fna + seqid2taxid.map (skip if already done,
#     e.g. a prior run of this script got partway through) ------------------

LIBRARY_FNA="$KRAKEN2_DB/library/library.fna"
SEQID_MAP="$KRAKEN2_DB/seqid2taxid.map"

if [ ! -s "$LIBRARY_FNA" ]; then
    echo "Decompressing $(ls "$FNA_DIR" | wc -l) genome files into one library FASTA (parallel across $NPROC workers)..."
    TMP_DIR="$KRAKEN2_DB/_bracken_prep_tmp"
    mkdir -p "$TMP_DIR"
    FILELIST="$TMP_DIR/fna_filelist.txt"
    ls "$FNA_DIR" > "$FILELIST"

    split -n "l/$NPROC" "$FILELIST" "$TMP_DIR/chunk_"

    pids=()
    for chunk in "$TMP_DIR"/chunk_*; do
        part="${chunk}.part"
        (
            : > "$part"
            while IFS= read -r f; do
                zcat "$FNA_DIR/$f" >> "$part"
            done < "$chunk"
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    : > "$LIBRARY_FNA.tmp"
    for chunk in "$TMP_DIR"/chunk_*; do
        [[ "$chunk" == *.part ]] && continue
        cat "${chunk}.part" >> "$LIBRARY_FNA.tmp"
    done
    mv "$LIBRARY_FNA.tmp" "$LIBRARY_FNA"
    rm -rf "$TMP_DIR"
    echo "Library built: $(du -h "$LIBRARY_FNA" | cut -f1)"
else
    echo "Library FASTA already exists at $LIBRARY_FNA -- skipping decompression."
fi

if [ ! -s "$SEQID_MAP" ]; then
    echo "Building seqid2taxid.map from library headers (full header token as key)..."
    grep "^>" "$LIBRARY_FNA" \
        | sed -E 's/^>(kraken:taxid\|([0-9]+)\|[^ ]+).*/\1\t\2/' \
        > "$SEQID_MAP.tmp"
    mv "$SEQID_MAP.tmp" "$SEQID_MAP"
    echo "seqid2taxid.map lines: $(wc -l < "$SEQID_MAP")"
    echo "Distinct genome taxids represented: $(awk -F'\t' '{print $2}' "$SEQID_MAP" | sort -u | wc -l)"
else
    echo "seqid2taxid.map already exists -- skipping."
fi

# --- Run bracken-build ---------------------------------------------------------
# (idempotent: skips regenerating database.kraken if it already exists, per
# the vendored script's own logic -- so a rerun after clearing a bad
# seqid2taxid.map only redoes the kmer2read_distr step, not the expensive
# self-classify pass)

echo "Running bracken-build (k=35, l=100, t=$NPROC)..."
apptainer exec --bind /rs1,/gpfs_common "$SIF" \
    bracken-build -d "$KRAKEN2_DB" -t "$NPROC" -k 35 -l 100

echo "Verifying: distinct genome_taxid values in the new kmer_distrib..."
awk -F'\t' 'NR>1{n=split($2,a," "); for(i=1;i<=n;i++){split(a[i],b,":"); print b[1]}}' \
    "$KRAKEN2_DB/database100mers.kmer_distrib" | sort -u | wc -l

echo "Job finished at $(date)"
