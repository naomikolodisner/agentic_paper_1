#!/usr/bin/env python3
"""
Idempotent, one-time extraction of a random subset of raw HumGut MAGs from
HumGut2.tar into one plain (ungzipped) FASTA per genome in
config.HUMGUT_RAW_MAGS_DIR.

HumGut2.tar has never been extracted -- PHORAGER hasn't run on HumGut yet, so
there is no prophage-stripped MAG set either. This pulls straight from the raw
tar (prophages intact) as a stand-in until PHORAGER output lands.

Skips entirely if HUMGUT_RAW_MAGS_DIR already has >= config.HUMGUT_SUBSET_N
files -- safe to call from generate_background.sh on every array task.
"""

import gzip
import random
import shutil
import sys
import tarfile
from pathlib import Path

import pandas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main():
    config.HUMGUT_RAW_MAGS_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(config.HUMGUT_RAW_MAGS_DIR.glob(config.HUMGUT_MAG_GLOB))
    if len(existing) >= config.HUMGUT_SUBSET_N:
        print(
            f"Already extracted ({len(existing)} files in "
            f"{config.HUMGUT_RAW_MAGS_DIR}) -- skipping."
        )
        return 0

    print(f"Reading {config.HUMGUT_TSV} ...")
    meta = pandas.read_csv(config.HUMGUT_TSV, sep="\t")

    random.seed(config.HUMGUT_SUBSET_SEED)
    chosen_files = random.sample(
        list(meta["genome_file"]), config.HUMGUT_SUBSET_N
    )
    members = [f"fna/{fname}" for fname in chosen_files]

    print(
        f"Extracting {len(members)} genomes from {config.HUMGUT_TAR} "
        f"(seed={config.HUMGUT_SUBSET_SEED}) ..."
    )

    with tarfile.open(config.HUMGUT_TAR, "r") as tar:
        for member_name in members:
            try:
                member = tar.getmember(member_name)
            except KeyError:
                print(f"WARNING: {member_name} not found in tar -- skipping", file=sys.stderr)
                continue

            genome_file = Path(member_name).name  # e.g. GUT_GENOME080427.fna.gz
            out_name = genome_file[: -len(".gz")] if genome_file.endswith(".gz") else genome_file
            out_name = Path(out_name).with_suffix(".fasta").name
            out_path = config.HUMGUT_RAW_MAGS_DIR / out_name

            if out_path.exists():
                continue

            src = tar.extractfile(member)
            if src is None:
                print(f"WARNING: could not read {member_name} -- skipping", file=sys.stderr)
                continue

            with gzip.open(src, "rb") as gz_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(gz_in, f_out)

    final_count = len(list(config.HUMGUT_RAW_MAGS_DIR.glob(config.HUMGUT_MAG_GLOB)))
    print(f"Done: {final_count} MAG FASTAs in {config.HUMGUT_RAW_MAGS_DIR}")

    if final_count == 0:
        print("ERROR: extraction produced zero MAG files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
