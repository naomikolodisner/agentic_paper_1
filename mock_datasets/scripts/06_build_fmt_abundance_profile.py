#!/usr/bin/env python
"""Build one ISS abundance file per FMT sample (genome path<TAB>abundance, no header).

Abundance is each representative genome's share of the sample's Bracken-corrected reads.
Genomes with no phoraged sequence are dropped and the rest renormalized.
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

representatives = pd.read_csv(config.OUT_REPRESENTATIVE_GENOMES, sep="\t", dtype={"taxid": str})
representatives["genome_id"] = representatives["genome_file"].str.replace(r"(\.fna)?(\.gz)?$", "", regex=True)
available = {p.stem for p in config.HUMGUT_PROPHAGE_REMOVED_DIR.glob("*.fna")}
config.OUT_FMT_ABUNDANCE_PROFILES.mkdir(parents=True, exist_ok=True)

for sample in args.sample_list.read_text().split():
    bracken = pd.read_csv(config.OUT_KRAKEN2_FMT_READS / sample / f"{sample}.bracken", sep="\t", dtype={"taxonomy_id": str})
    profile = (
        bracken[bracken["new_est_reads"] > 0]
        .merge(representatives, left_on="taxonomy_id", right_on="taxid")
        .loc[lambda d: d["genome_id"].isin(available)]
        .groupby("genome_id", as_index=False)["new_est_reads"].sum()
    )
    profile["abundance"] = profile["new_est_reads"] / profile["new_est_reads"].sum()
    profile["genome_path"] = [str((config.HUMGUT_PROPHAGE_REMOVED_DIR / f"{g}.fna").resolve()) for g in profile["genome_id"]]

    out = config.OUT_FMT_ABUNDANCE_PROFILES / f"{sample}_abundance.tsv"
    profile.sort_values("abundance", ascending=False)[["genome_path", "abundance"]].to_csv(out, sep="\t", index=False, header=False)
    print(f"{sample}: {len(profile)} genomes -> {out}")
