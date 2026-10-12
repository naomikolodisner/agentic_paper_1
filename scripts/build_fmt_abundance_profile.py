#!/usr/bin/env python
"""
build_fmt_abundance_profile.py

Builds one ISS-shaped abundance profile per real FMT sample
(config.FMT_READS_SAMPLE_LIST), replacing the old built-in lognormal/
exponential ISS abundance distributions with real per-genome sequencing
depth derived from that sample's own Bracken-corrected kraken2 classification.

For each sample:
    1. Load its .bracken output (config.OUT_KRAKEN2_FMT_READS/<sample>/) --
       new_est_reads is Bracken's corrected per-taxid read count (see
       NOTEBOOK.md 2026-09-20: raw kraken2 counts alone undercount any taxon
       with close relatives in HumGut, since ambiguous reads get pushed up
       to a shared ancestor instead of a specific genome).
    2. Join each taxid to its representative genome file
       (config.OUT_REPRESENTATIVE_GENOMES, built by
       scripts/pick_representative_genomes.py -- picks the candidate with
       the highest prevalence_score when a taxid has more than one HumGut
       genome file mapped to it).
    3. Sum new_est_reads per representative genome, normalize to a fraction
       of that sample's total.
    4. Write "genome_path<TAB>abundance" to
       config.OUT_FMT_ABUNDANCE_PROFILES/<sample>_abundance.tsv -- genome_path
       is the real, absolute path to that genome's extracted phoraged FASTA
       under config.HUMGUT_PROPHAGE_REMOVED_DIR (scripts/extract_humgut_phoraged.py
       must have already run). Verified directly in ISS's own source
       (iss/generator.py + iss/abundance.py) that when --abundance_file is
       combined with --draft, ISS matches the abundance file's genome_id
       column against the LITERAL file path string passed to --draft -- not
       a basename, not the FASTA record ID inside the file -- so this file
       is only usable if whatever calls `iss generate --draft` passes
       EXACTLY these same path strings. Safest way to guarantee that: derive
       the --draft argument list FROM this file's own first column (e.g.
       `cut -f1 <profile>.tsv`) rather than reconstructing paths
       independently in two places that could silently drift apart.

Usage:
    python scripts/build_fmt_abundance_profile.py [--sample-list PATH]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def strip_genome_extension(genome_file: str) -> str:
    """'GUT_GENOME080427.fna.gz' -> 'GUT_GENOME080427'"""
    name = genome_file
    for ext in (".gz", ".fna", ".fa", ".fasta"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def build_profile(sample: str, representatives: pd.DataFrame, available_genome_ids: set[str]) -> pd.DataFrame:
    bracken_file = config.OUT_KRAKEN2_FMT_READS / sample / f"{sample}.bracken"
    if not bracken_file.exists():
        raise FileNotFoundError(f"No .bracken file for {sample} at {bracken_file}")

    bracken = pd.read_csv(bracken_file, sep="\t", dtype={"taxonomy_id": str})
    bracken = bracken.loc[bracken["new_est_reads"] > 0, ["taxonomy_id", "new_est_reads"]]

    merged = bracken.merge(
        representatives[["taxid", "genome_file"]],
        left_on="taxonomy_id",
        right_on="taxid",
        how="left",
    )

    unmatched = merged["genome_file"].isna().sum()
    if unmatched:
        print(
            f"WARNING: {sample}: {unmatched} taxid(s) with reads have no representative "
            f"genome (run scripts/pick_representative_genomes.py first) -- dropped from profile.",
            file=sys.stderr,
        )
    merged = merged.dropna(subset=["genome_file"])

    merged["genome_id"] = merged["genome_file"].map(strip_genome_extension)
    profile = merged.groupby("genome_id", as_index=False)["new_est_reads"].sum()

    # PHORAGER only processed 30691/31225 (98.3%) of HumGut (verified 2026-09-21 by
    # counting members across all 7 HumGut_phoraged_*.tar.gz shards) -- a genome can
    # be a valid representative with real Bracken-corrected reads but still have no
    # phoraged sequence available at all. Drop those and renormalize rather than
    # silently reference a --draft file that will never exist, or fall back to an
    # intact-prophage sequence for methodological consistency. Checked the actual
    # magnitude before deciding this was safe: <=0.00007 of total abundance mass
    # per sample (14-22 genomes out of ~2300-3100 per profile).
    n_before = len(profile)
    reads_before = profile["new_est_reads"].sum()
    profile = profile[profile["genome_id"].isin(available_genome_ids)]
    dropped = n_before - len(profile)
    if dropped:
        dropped_fraction = 1 - (profile["new_est_reads"].sum() / reads_before)
        print(
            f"{sample}: dropped {dropped} genome(s) with no phoraged sequence available "
            f"({dropped_fraction:.5f} of total abundance mass) -- renormalizing remainder.",
            file=sys.stderr,
        )

    profile["abundance"] = profile["new_est_reads"] / profile["new_est_reads"].sum()
    profile["genome_path"] = profile["genome_id"].map(
        lambda gid: str((config.HUMGUT_PROPHAGE_REMOVED_DIR / f"{gid}.fna").resolve())
    )
    return profile[["genome_path", "abundance"]].sort_values("abundance", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=config.FMT_READS_SAMPLE_LIST,
        help="Sample list to build profiles for (default: config.FMT_READS_SAMPLE_LIST)",
    )
    args = parser.parse_args()

    if not config.OUT_REPRESENTATIVE_GENOMES.exists():
        sys.exit(
            f"ERROR: {config.OUT_REPRESENTATIVE_GENOMES} not found -- "
            f"run scripts/pick_representative_genomes.py first."
        )
    representatives = pd.read_csv(config.OUT_REPRESENTATIVE_GENOMES, sep="\t", dtype={"taxid": str})

    if not config.HUMGUT_PROPHAGE_REMOVED_DIR.exists():
        sys.exit(
            f"ERROR: {config.HUMGUT_PROPHAGE_REMOVED_DIR} not found -- "
            f"run scripts/extract_humgut_phoraged.py first."
        )
    available_genome_ids = {
        p.stem for p in config.HUMGUT_PROPHAGE_REMOVED_DIR.glob("*.fna")
    }

    samples = [line.strip() for line in args.sample_list.read_text().splitlines() if line.strip()]
    config.OUT_FMT_ABUNDANCE_PROFILES.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        profile = build_profile(sample, representatives, available_genome_ids)
        out_path = config.OUT_FMT_ABUNDANCE_PROFILES / f"{sample}_abundance.tsv"
        profile.to_csv(out_path, sep="\t", index=False, header=False)
        print(f"{sample}: {len(profile)} genomes, wrote {out_path}")


if __name__ == "__main__":
    main()
