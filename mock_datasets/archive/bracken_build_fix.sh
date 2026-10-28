#!/usr/bin/env bash
# =============================================================================
# bracken_build_fix.sh
#
# Fixes a bug in the first mock_datasets/scripts/02_bracken_build.sh run (job 888905):
# seqid2taxid.map was built using each genome's STRIPPED sequence ID (e.g.
# "HumGut_17319_1"), assuming kraken2-build's ">kraken:taxid|N|seqid"
# convention gets parsed and stripped during plain `kraken2` classify the
# same way it is during `kraken2-build --add-to-library`. Verified this is
# wrong by reading data/humgut_kraken2/database.kraken directly: kraken2
# classify keeps the WHOLE first header token -- including the
# "kraken:taxid|N|" prefix -- as the sequence ID (that stripping only
# happens at library-build time, not at classify time). Because
# seqid2taxid.map's keys never matched any real sequence ID,
# kmer2read_distr's --seqid2taxid lookups all failed silently and it fell
# back to one constant bogus "genome" (273997528, not even in our 4xxxxxx
# custom taxid range) for every single sequence -- confirmed by inspecting
# database100mers.kraken and database100mers.kmer_distrib directly, both of
# which showed that one fake value on every line. Running `bracken` against
# this corrupted table produced "Reads distributed: 0" for all 3 test
# samples -- Bracken silently did nothing beyond thresholding.
#
# Fix: regenerate seqid2taxid.map with the FULL header token as the key
# (validated against real fna headers before running this at scale). Then
# re-run bracken-build -- it explicitly skips regenerating database.kraken
# if it already exists (see mock_datasets/scripts/02_bracken_build.sh's copy of the vendored
# script logic), so this does NOT redo the expensive 41-minute self-classify
# pass, only the kmer2read_distr + generate_kmer_distribution.py step that
# was actually broken.
#
# Usage:
#   sbatch mock_datasets/scripts/bracken_build_fix.sh
# =============================================================================
#SBATCH --partition=compute
#SBATCH --job-name=bracken_fix
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.out
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=nkolodi@ncsu.edu

echo "Job started at $(date) on $(hostname)"
set -euo pipefail

KRAKEN2_DB="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/data/humgut_kraken2"
SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_bracken:3.1--h9948957_0.sif"
NPROC="${SLURM_CPUS_PER_TASK:-10}"

module load apptainer/1.4.2-1

echo "Regenerating seqid2taxid.map with the full header token as key..."
grep "^>" "$KRAKEN2_DB/library/library.fna" \
    | sed -E 's/^>(kraken:taxid\|([0-9]+)\|[^ ]+).*/\1\t\2/' \
    > "$KRAKEN2_DB/seqid2taxid.map.tmp"
mv "$KRAKEN2_DB/seqid2taxid.map.tmp" "$KRAKEN2_DB/seqid2taxid.map"
echo "New seqid2taxid.map lines: $(wc -l < "$KRAKEN2_DB/seqid2taxid.map")"
echo "Distinct genome taxids represented: $(awk -F'\t' '{print $2}' "$KRAKEN2_DB/seqid2taxid.map" | sort -u | wc -l)"

echo "Removing stale (corrupted) database100mers.* outputs..."
rm -f "$KRAKEN2_DB/database100mers.kraken" "$KRAKEN2_DB/database100mers.kmer_distrib"

echo "Re-running bracken-build (should skip the self-classify step, database.kraken already exists)..."
apptainer exec --bind /rs1,/gpfs_common "$SIF" \
    bracken-build -d "$KRAKEN2_DB" -t "$NPROC" -k 35 -l 100

echo "Verifying: distinct genome_taxid values in the new kmer_distrib..."
awk -F'\t' 'NR>1{n=split($2,a," "); for(i=1;i<=n;i++){split(a[i],b,":"); print b[1]}}' \
    "$KRAKEN2_DB/database100mers.kmer_distrib" | sort -u | wc -l

echo "Job finished at $(date)"
