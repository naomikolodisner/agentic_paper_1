#!/usr/bin/env python3
"""
Cross-checks every accession in the current full INPHARED catalog
(/rs1/shares/brc/databases/inphared/all_genomes.txt, ~2902 phage genomes)
against kraken2_pluspfp's seqid2taxid.map, so pick_refs.py's spike-in pool
only ever draws genomes kraken2_check.sh can actually confirm as detectable.

Supersedes the archived scripts/archive/check_kraken2_accessions.sh, which
did the same cross-check but against the older, much smaller
106-accession INPHARED/VirFinder pre-2014 overlap list (grep-per-accession
over the map -- fine at 106 accessions, too slow at ~2900). This version
builds the set of kraken2-covered accessions in one pass over the map
instead.

Usage:
    python scripts/check_kraken2_accessions.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ALL_GENOMES_LIST = Path("/rs1/shares/brc/databases/inphared/all_genomes.txt")
SEQID2TAXID = config.KRAKEN2_SEQID2TAXID  # kraken2_pluspfp/seqid2taxid.map

OUTPUT_FOUND = config.INPHARED_ACCESSIONS_LIST
OUTPUT_MISSING = OUTPUT_FOUND.parent / "accessions_missing_from_kraken2.txt"

ACCESSION_IN_MAP = re.compile(r"\|([A-Za-z_]+\d+)\.\d+")


def main():
    kraken_accessions = set()
    with open(SEQID2TAXID) as f:
        for line in f:
            m = ACCESSION_IN_MAP.search(line)
            if m:
                kraken_accessions.add(m.group(1))

    with open(ALL_GENOMES_LIST) as f:
        all_accessions = [
            line.strip().removesuffix(".fasta") for line in f if line.strip()
        ]

    found = [a for a in all_accessions if a in kraken_accessions]
    missing = [a for a in all_accessions if a not in kraken_accessions]

    OUTPUT_FOUND.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FOUND, "w") as f:
        f.write("\n".join(found) + "\n")
    with open(OUTPUT_MISSING, "w") as f:
        f.write("\n".join(missing) + "\n")

    print(f"Checked {len(all_accessions)} INPHARED accessions against {SEQID2TAXID}")
    print(f"  Found (kraken2-detectable):   {len(found)} -> {OUTPUT_FOUND}")
    print(f"  Missing (not in kraken2 DB):  {len(missing)} -> {OUTPUT_MISSING}")


if __name__ == "__main__":
    main()
