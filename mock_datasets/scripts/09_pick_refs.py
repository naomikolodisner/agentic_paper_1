#!/usr/bin/env python3

# randomly picks which viral references get spiked into each sample, and
# writes each sample dir's conditions.tsv (which per-genome ART coverages
# 10_read_mixing.sh should build for it)
#
# Two designs:
#   --design og       (default) Replica of the PI's original ART scripts:
#                     SPIKE_IN_DIR/{single,equal2,equal3,equal4,unequal2,unequal3}/sample{1..15}/
#                     with config.SPIKE_OG_DESIGNS' conditions.
#   --design subsets  Large-subset design: SPIKE_IN_DIR/sample_{size}v_{n}/ for
#                     config.VIRAL_SUBSET_SIZES, every genome at the same
#                     coverage, one condition per config.SPIKE_COVERAGES value.
#
# Output per sample dir: contigs.fasta, refs_log.txt, conditions.tsv
# (condition<TAB>comma-separated per-genome coverages, in contigs.fasta order).

import argparse
import random
import sys
from pathlib import Path
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--design", choices=["og", "subsets"], default="og")
args = parser.parse_args()

# choose a random seed to reproduce this later
random.seed(41)

# This is the list of reference genomes we can choose from -- the 408
# INPHARED accessions confirmed present in kraken2_pluspfp's DB
# (config.INPHARED_ACCESSIONS_LIST). Each accession is its own single-record
# FASTA file in config.INPHARED_GENOMES_DIR.
with open(config.INPHARED_ACCESSIONS_LIST) as f:
    contig_ids = [line.strip() for line in f if line.strip()]

config.SPIKE_IN_DIR.mkdir(parents=True, exist_ok=True)

# (out_dir, n_viruses, {condition: [per-genome coverages]})
samples = []
if args.design == "og":
    for design_name, design in config.SPIKE_OG_DESIGNS.items():
        for replicate in range(config.SPIKE_OG_REPLICATES):
            out_dir = config.SPIKE_IN_DIR / design_name / f"sample{replicate + 1}"
            samples.append((out_dir, design["n_viruses"], design["conditions"]))
else:
    for subset_size in config.VIRAL_SUBSET_SIZES:
        for replicate in range(config.NUM_REPLICATES_PER_SUBSET_SIZE):
            out_dir = config.SPIKE_IN_DIR / f"sample_{subset_size}v_{replicate + 1}"
            conditions = {f"{c:g}x": [c] * subset_size for c in config.SPIKE_COVERAGES}
            samples.append((out_dir, subset_size, conditions))

for out_dir, n_viruses, conditions in samples:
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = random.sample(contig_ids, n_viruses)

    with open(out_dir / "refs_log.txt", "w") as log_f:
        for ref in refs:
            log_f.write(f"{ref}\n")

    out_fasta = out_dir / "contigs.fasta"
    with open(out_fasta, "w") as f:
        for ref in refs:
            genome_file = config.INPHARED_GENOMES_DIR / f"{ref}.fasta"
            if genome_file.exists():
                record = next(SeqIO.parse(str(genome_file), "fasta"))
                SeqIO.write(record, f, "fasta")
            else:
                sys.exit(f"ERROR: {ref} not found in {config.INPHARED_GENOMES_DIR}")

    with open(out_dir / "conditions.tsv", "w") as cond_f:
        for name, coverages in conditions.items():
            assert len(coverages) == n_viruses, (name, coverages)
            cond_f.write(f"{name}\t{','.join(f'{c:g}' for c in coverages)}\n")

print(f"Wrote {len(samples)} sample dirs ({args.design} design) under {config.SPIKE_IN_DIR}")
