#!/usr/bin/env python
"""Extract the representative genomes (04) from one phoraged HumGut tarball shard, as <genome_id>.fna.

The shards name genomes by HumGut_name; HumGut2.tsv maps those to the genome_file names 04 uses.
Usage: python extract_humgut_phoraged.py --shard-index 0   # 0-6
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--shard-index", type=int, required=True)
args = parser.parse_args()

shard = sorted(PHORAGED_DIR.glob("HumGut_phoraged_*.tar.gz"))[args.shard_index]
representatives = pd.read_csv(config.OUT_REPRESENTATIVE_GENOMES, sep="\t")
humgut = pd.read_csv(config.HUMGUT_TSV, sep="\t", usecols=["HumGut_name", "genome_file"])
needed = dict(representatives.merge(humgut, on="genome_file")[["HumGut_name", "genome_file"]].values)

config.HUMGUT_PROPHAGE_REMOVED_DIR.mkdir(parents=True, exist_ok=True)
n = 0
# Stream mode ("r|gz"): one pass through the shard instead of decompressing it once per member
with tarfile.open(shard, "r|gz") as tar:
    for member in tar:
        humgut_name = Path(member.name).name.removesuffix(SUFFIX)
        if humgut_name not in needed:
            continue
        genome_id = needed[humgut_name].removesuffix(".gz").removesuffix(".fna")
        out = config.HUMGUT_PROPHAGE_REMOVED_DIR / f"{genome_id}.fna"
        if out.exists():
            continue
        with gzip.open(tar.extractfile(member), "rt") as fin, open(out, "w") as fout:
            shutil.copyfileobj(fin, fout)
        n += 1

print(f"{shard.name}: extracted {n} genomes")
