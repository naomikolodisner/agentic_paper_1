#!/usr/bin/env bash
# Build Bracken's kmer distribution (read length 100) for the HumGut kraken2 DB.
# bracken-build needs library/, taxonomy/ and seqid2taxid.map, which this DB doesn't keep, so they are rebuilt first.
# Usage: sbatch mock_datasets/scripts/02_bracken_build.sh
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

set -euo pipefail

PROJECT="/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1"
PYTHON_BIN="/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python"
SIF="/rs1/shares/brc/admin/containers/images/quay.io_biocontainers_bracken:3.1--h9948957_0.sif"
cfg() { "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$PROJECT'); import config; print(config.$1)"; }

DB=$(cfg HUMGUT_KRAKEN2_DB)
READ_LEN=100
module load apptainer/1.4.2-1

mkdir -p "$DB/library" "$DB/taxonomy"
ln -sf "$DB/GTDB_nodes.dmp" "$DB/taxonomy/nodes.dmp"
ln -sf "$DB/GTDB_names.dmp" "$DB/taxonomy/names.dmp"

# One concatenated library file: bracken-build passes every library file to kraken2 as its own argument, and ~31k genome paths exceed ARG_MAX
if [ ! -s "$DB/library/library.fna" ]; then
    find -L "$DB/fna" -name '*.fna.gz' -exec zcat {} + > "$DB/library/library.fna.tmp"
    mv "$DB/library/library.fna.tmp" "$DB/library/library.fna"
fi

# Keyed on the whole "kraken:taxid|<taxid>|<seqid>" token, since that is the sequence ID kraken2 reports
grep '^>' "$DB/library/library.fna" \
    | sed -E 's/^>(kraken:taxid\|([0-9]+)\|[^ ]+).*/\1\t\2/' \
    > "$DB/seqid2taxid.map"

apptainer exec --bind /rs1,/gpfs_common "$SIF" \
    bracken-build -d "$DB" -t "${SLURM_CPUS_PER_TASK:-10}" -k 35 -l "$READ_LEN"
