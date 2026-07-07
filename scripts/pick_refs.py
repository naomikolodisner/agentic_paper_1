#!/usr/bin/env python3

# randomly picks which viral references get spiked into each sample

import random
import sys
from pathlib import Path
from Bio import SeqIO
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# choose a random seed to reproduce this later
random.seed(41)

# This is the list of reference genomes we can choose from
references_meta = pandas.read_csv(config.AVRC_METADATA_CSV, sep=',', comment='#')
contig_ids = list(references_meta['contig_id'])

fasta_index = SeqIO.index(str(config.AVRC_ALL_SEQUENCES), "fasta")

config.SPIKE_IN_DIR.mkdir(parents=True, exist_ok=True)

for subset_size in config.VIRAL_SUBSET_SIZES:
    for replicate in range(config.NUM_REPLICATES_PER_SUBSET_SIZE):
        out_dir = config.SPIKE_IN_DIR / f"sample_{subset_size}v_{replicate + 1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        refs = random.sample(contig_ids, subset_size)

        with open(out_dir / "refs_log.txt", "w") as log_f:
            for ref in refs:
                log_f.write(f"{ref}\n")

        out_fasta = out_dir / "contigs.fasta"
        with open(out_fasta, "w") as f:
            for ref in refs:
                if ref in fasta_index:
                    SeqIO.write(fasta_index[ref], f, "fasta")
                else:
                    print(f"Warning: {ref} not found in {config.AVRC_ALL_SEQUENCES}")
