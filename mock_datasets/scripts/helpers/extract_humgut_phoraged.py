#!/usr/bin/env python
"""
extract_humgut_phoraged.py

Pulls the specific phoraged (prophage-stripped) HumGut genomes our abundance
profiles actually need (config.OUT_REPRESENTATIVE_GENOMES, ~3455 genomes)
out of the shared HumGut_phoraged_*.tar.gz shards at
/rs1/shares/brc/admin/databases/humgut_phoraged/ -- those shards hold the
*entire* HumGut collection (~31225 genomes across 7 archives), each inner
file individually gzipped, so this does a single sequential pass per shard
(tarfile stream mode -- avoids the double-decompress-the-whole-archive
gotcha of calling getmembers() then extractfile() member-by-member) and only
writes out the ones we need.

Naming mismatch handled here: the archives key files by `HumGut_name`
(e.g. "HumGut_2753_phoraged.fasta.gz"), but our representative-genome table
identifies genomes by the `genome_file` column (e.g.
"GUT_GENOME080427.fna.gz") -- joined through HumGut2.tsv, which has both.

One array task per shard (see mock_datasets/scripts/05_extract_humgut_phoraged.sh) --
--shard-index selects which of the 7 shards (sorted by filename) this task
scans. All tasks write into the same shared output dir
(config.HUMGUT_PROPHAGE_REMOVED_DIR), one genome per file, renamed to
"<genome_id>.fna" where genome_id is the bare accession used everywhere else
in this pipeline (config.OUT_FMT_ABUNDANCE_PROFILES's genome_id column,
mock_datasets/scripts/06_build_fmt_abundance_profile.py's strip_genome_extension()) -- so a
later background-generation step can resolve genome_id -> real path with a
plain glob against this directory.

Usage:
    python mock_datasets/scripts/helpers/extract_humgut_phoraged.py --shard-index 0   # one of 7 shards, 0-6
"""
import argparse
import gzip
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config

PHORAGED_DIR = Path("/rs1/shares/brc/admin/databases/humgut_phoraged")
SUFFIX = "_phoraged.fasta.gz"


def strip_genome_extension(genome_file: str) -> str:
    """'GUT_GENOME080427.fna.gz' -> 'GUT_GENOME080427'"""
    name = genome_file
    for ext in (".gz", ".fna", ".fa", ".fasta"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def load_needed_genomes() -> dict[str, str]:
    """Returns {HumGut_name: genome_file} for every genome our profiles need."""
    if not config.OUT_REPRESENTATIVE_GENOMES.exists():
        sys.exit(
            f"ERROR: {config.OUT_REPRESENTATIVE_GENOMES} not found -- "
            f"run mock_datasets/scripts/04_pick_representative_genomes.py first."
        )
    representatives = pd.read_csv(config.OUT_REPRESENTATIVE_GENOMES, sep="\t")
    manifest = pd.read_csv(config.HUMGUT_TSV, sep="\t")

    merged = representatives.merge(
        manifest[["HumGut_name", "genome_file"]], on="genome_file", how="left"
    )
    missing = merged["HumGut_name"].isna().sum()
    if missing:
        print(
            f"WARNING: {missing} representative genome(s) have no HumGut_name match in "
            f"{config.HUMGUT_TSV} -- dropped.",
            file=sys.stderr,
        )
    merged = merged.dropna(subset=["HumGut_name"])
    return dict(zip(merged["HumGut_name"], merged["genome_file"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True, help="0-indexed shard to process (sorted by filename)")
    args = parser.parse_args()

    shards = sorted(PHORAGED_DIR.glob("HumGut_phoraged_*.tar.gz"))
    if args.shard_index >= len(shards):
        sys.exit(f"ERROR: shard index {args.shard_index} out of range ({len(shards)} shards found)")
    shard = shards[args.shard_index]

    needed = load_needed_genomes()
    print(f"Shard: {shard.name}")
    print(f"Need to find {len(needed)} genomes total across all shards (this task scans only this one).")

    config.HUMGUT_PROPHAGE_REMOVED_DIR.mkdir(parents=True, exist_ok=True)

    found_this_shard = 0
    skipped_existing = 0
    with tarfile.open(shard, "r|gz") as tf:
        for member in tf:
            base = Path(member.name).name
            if not base.endswith(SUFFIX):
                continue
            humgut_name = base[: -len(SUFFIX)]
            if humgut_name not in needed:
                continue

            genome_id = strip_genome_extension(needed[humgut_name])
            out_path = config.HUMGUT_PROPHAGE_REMOVED_DIR / f"{genome_id}.fna"
            if out_path.exists():
                skipped_existing += 1
                continue

            f = tf.extractfile(member)
            with gzip.open(f, "rt") as gz_in, open(out_path, "w") as out:
                shutil.copyfileobj(gz_in, out)
            found_this_shard += 1

    print(
        f"Done with {shard.name}: extracted {found_this_shard} new genomes, "
        f"{skipped_existing} already present from a prior run."
    )


if __name__ == "__main__":
    main()
