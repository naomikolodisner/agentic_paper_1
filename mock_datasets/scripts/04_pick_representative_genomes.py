#!/usr/bin/env python
"""Pick one HumGut genome per taxid seen in any sample's Bracken output (highest prevalence_score).

Several HumGut genomes can share one taxid, but ISS needs one genome file per organism.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample-list", type=Path, default=config.FMT_READS_SAMPLE_LIST)
args = parser.parse_args()

taxids = set()
for sample in args.sample_list.read_text().split():
    bracken = pd.read_csv(config.OUT_KRAKEN2_FMT_READS / sample / f"{sample}.bracken", sep="\t", dtype={"taxonomy_id": str})
    taxids.update(bracken.loc[bracken["new_est_reads"] > 0, "taxonomy_id"])

humgut = pd.read_csv(config.HUMGUT_TSV, sep="\t", dtype={"gtdb_tax_id": str})
humgut = humgut[humgut["gtdb_tax_id"].isin(taxids)]
humgut["n_candidates"] = humgut.groupby("gtdb_tax_id")["genome_file"].transform("size")

# Highest prevalence_score wins; ties broken on genome_file so reruns give the same pick
best = (
    humgut.sort_values(["prevalence_score", "genome_file"], ascending=[False, True])
    .drop_duplicates("gtdb_tax_id")
    .rename(columns={"gtdb_tax_id": "taxid"})
    .sort_values("taxid")
)
best[["taxid", "gtdb_name", "genome_file", "prevalence_score", "n_candidates"]].to_csv(
    config.OUT_REPRESENTATIVE_GENOMES, sep="\t", index=False
)
print(f"{len(taxids)} taxids, {len(best)} representative genomes -> {config.OUT_REPRESENTATIVE_GENOMES}")
