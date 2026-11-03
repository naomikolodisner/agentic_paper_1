#!/usr/bin/env python3
"""Pick random spike-in genomes for each sample dir and write its conditions.

Each sample dir gets contigs.fasta, refs_log.txt and conditions.tsv
(condition<TAB>comma-separated per-genome coverages, in contigs.fasta order).
  --design og       {type}/sample{n}/ for each type in config.SPIKE_OG_DESIGNS
  --design subsets  sample_{size}v_{n}/ for config.VIRAL_SUBSET_SIZES, one condition per config.SPIKE_COVERAGES
"""
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

random.seed(41)
accessions = config.INPHARED_ACCESSIONS_LIST.read_text().split()

# (out_dir, n_viruses, {condition: [per-genome coverages]})
samples = []
if args.design == "og":
    for name, design in config.SPIKE_OG_DESIGNS.items():
        for rep in range(1, config.SPIKE_OG_REPLICATES + 1):
            samples.append((config.SPIKE_IN_DIR / name / f"sample{rep}", design["n_viruses"], design["conditions"]))
else:
    for size in config.VIRAL_SUBSET_SIZES:
        for rep in range(1, config.NUM_REPLICATES_PER_SUBSET_SIZE + 1):
            conditions = {f"{c:g}x": [c] * size for c in config.SPIKE_COVERAGES}
            samples.append((config.SPIKE_IN_DIR / f"sample_{size}v_{rep}", size, conditions))

for out_dir, n_viruses, conditions in samples:
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = random.sample(accessions, n_viruses)

    (out_dir / "refs_log.txt").write_text("".join(f"{r}\n" for r in refs))
    records = [next(SeqIO.parse(str(config.INPHARED_GENOMES_DIR / f"{r}.fasta"), "fasta")) for r in refs]
    SeqIO.write(records, str(out_dir / "contigs.fasta"), "fasta")

    with open(out_dir / "conditions.tsv", "w") as f:
        for name, coverages in conditions.items():
            assert len(coverages) == n_viruses, (name, coverages)
            f.write(f"{name}\t{','.join(f'{c:g}' for c in coverages)}\n")

print(f"Wrote {len(samples)} sample dirs ({args.design} design) under {config.SPIKE_IN_DIR}")
