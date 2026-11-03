#!/usr/bin/env python3
"""List which INPHARED accessions are in the kraken2 DB's seqid2taxid.map (the spike-in genome pool)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

ALL_GENOMES_LIST = Path("/rs1/shares/brc/databases/inphared/all_genomes.txt")
OUT_FOUND = config.INPHARED_ACCESSIONS_LIST
OUT_MISSING = OUT_FOUND.parent / "accessions_missing_from_kraken2.txt"

accession = re.compile(r"\|([A-Za-z_]+\d+)\.\d+")
with open(config.KRAKEN2_SEQID2TAXID) as f:
    in_kraken = {m.group(1) for line in f if (m := accession.search(line))}

all_accessions = [a.removesuffix(".fasta") for a in ALL_GENOMES_LIST.read_text().split()]
found = [a for a in all_accessions if a in in_kraken]
missing = [a for a in all_accessions if a not in in_kraken]

OUT_FOUND.write_text("\n".join(found) + "\n")
OUT_MISSING.write_text("\n".join(missing) + "\n")
print(f"{len(all_accessions)} accessions: {len(found)} in kraken2 -> {OUT_FOUND}, {len(missing)} missing -> {OUT_MISSING}")
