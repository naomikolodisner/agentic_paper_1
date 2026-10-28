#!/usr/bin/env python
"""
04_pick_representative_genomes.py

Our HumGut kraken2 DB (config.HUMGUT_KRAKEN2_DB) assigns each genome a taxid
equal to HumGut's own `gtdb_tax_id` column (GTDB species-level) -- verified
2026-09-20 (4940 distinct gtdb_tax_id values in HumGut2.tsv vs. 4939 distinct
genome taxids in the fixed Bracken index). That's coarser than HumGut's own
genome-level `cluster975` clustering, so several genome files (cluster975
entries) can share one taxid. Bracken's corrected .bracken output gives us a
read count per TAXID, but building an ISS background needs one concrete
genome FASTA per organism -- so for any taxid with more than one candidate
genome file, we need to pick one to stand in for the whole cluster.

Picks by highest `prevalence_score` (how consistently that specific genome
was detected across HumGut's reference metagenomes) among the candidate
genome_file rows sharing a taxid, per direction from the user 2026-09-21 --
chosen over a dRep-style completeness/contamination quality score because
the goal here is a realistic background community (what's actually commonly
present in real gut samples), not the cleanest possible reference assembly.

Usage:
    python mock_datasets/scripts/04_pick_representative_genomes.py [--sample-list PATH]

Reads every sample's .bracken file under config.OUT_KRAKEN2_FMT_READS (for
samples in config.FMT_READS_SAMPLE_LIST, or --sample-list) to collect the
union of taxids with nonzero corrected abundance across all samples, then
writes one representative-genome row per taxid to
config.OUT_REPRESENTATIVE_GENOMES.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def load_observed_taxids(sample_list_path: Path) -> set[str]:
    samples = [
        line.strip()
        for line in sample_list_path.read_text().splitlines()
        if line.strip()
    ]
    taxids: set[str] = set()
    for sample in samples:
        bracken_file = config.OUT_KRAKEN2_FMT_READS / sample / f"{sample}.bracken"
        if not bracken_file.exists():
            print(f"WARNING: no .bracken file for {sample} at {bracken_file} -- skipping", file=sys.stderr)
            continue
        df = pd.read_csv(bracken_file, sep="\t", dtype={"taxonomy_id": str})
        taxids.update(df.loc[df["new_est_reads"] > 0, "taxonomy_id"])
    return taxids


def pick_representatives(taxids: set[str]) -> pd.DataFrame:
    manifest = pd.read_csv(config.HUMGUT_TSV, sep="\t", dtype={"gtdb_tax_id": str})

    rows = []
    missing = []
    for taxid in sorted(taxids):
        candidates = manifest.loc[manifest["gtdb_tax_id"] == taxid]
        if candidates.empty:
            missing.append(taxid)
            continue
        # Highest prevalence_score wins; break ties deterministically on
        # genome_file so re-runs are reproducible.
        best = candidates.sort_values(
            ["prevalence_score", "genome_file"], ascending=[False, True]
        ).iloc[0]
        rows.append(
            {
                "taxid": taxid,
                "gtdb_name": best["gtdb_name"],
                "genome_file": best["genome_file"],
                "prevalence_score": best["prevalence_score"],
                "n_candidates": len(candidates),
            }
        )

    if missing:
        print(
            f"WARNING: {len(missing)} taxid(s) observed in Bracken output have no "
            f"matching row in {config.HUMGUT_TSV} -- this shouldn't happen since our "
            f"kraken2 DB was built directly from this manifest. Missing taxids: "
            f"{missing[:20]}{' ...' if len(missing) > 20 else ''}",
            file=sys.stderr,
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=config.FMT_READS_SAMPLE_LIST,
        help="Sample list to pull observed taxids from (default: config.FMT_READS_SAMPLE_LIST)",
    )
    args = parser.parse_args()

    taxids = load_observed_taxids(args.sample_list)
    print(f"Observed {len(taxids)} distinct taxids with nonzero corrected abundance across all samples.")

    result = pick_representatives(taxids)
    config.OUT_REPRESENTATIVE_GENOMES.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.OUT_REPRESENTATIVE_GENOMES, sep="\t", index=False)

    n_multi = (result["n_candidates"] > 1).sum()
    print(f"Wrote {len(result)} representative genomes to {config.OUT_REPRESENTATIVE_GENOMES}")
    print(f"  {n_multi} taxid(s) had more than one candidate genome file (prevalence_score used to pick).")
    print(f"  {len(result) - n_multi} taxid(s) had only one candidate (no choice needed).")


if __name__ == "__main__":
    main()
