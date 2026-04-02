#!/usr/bin/env python3

# randomly picks which references get spiked into each sample

import random
import subprocess
from pathlib import Path
from Bio import SeqIO
import pandas

# This is the number of samples we are generating
NUM_SAMPLES = 15

# choose a random seed to reproduce this later
random.seed(41)

# This is the list of reference genomes we can choose from
references_meta = pandas.read_csv(
    '/groups/gwatts/databases/AVrC/database_csv/AvRCv1.Merged_ViralDesc.csv', sep=',', comment='#')
contig_ids = list(references_meta['contig_id'])

fasta_file = "/groups/gwatts/databases/AVrC/AVrC_allsequences.fasta"
fasta_index = SeqIO.index(fasta_file, "fasta")

ref_outdir = Path("/xdisk/gwatts/kolodisner/agentic_paper_1/dataset_creation/samples")

sample_types = {
    'single': 1,
    'equal2': 2,
    'equal3': 3,
    'equal4': 4,
    'unequal2': 2,
    'unequal3': 3,
}

for stype, num_refs in sample_types.items():
    for i in range(NUM_SAMPLES):
        out_dir = ref_outdir / f"{stype}/sample{i+1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        refs = random.sample(contig_ids, num_refs)

        with open(out_dir / "refs_log.txt", "w") as log_f:
            for ref in refs:
                log_f.write(f"{ref}\n")

        out_fasta = out_dir / "contigs.fasta"
        with open(out_fasta, "w") as f:
            for ref in refs:
                if ref in fasta_index:
                    SeqIO.write(fasta_index[ref], f, "fasta")
                else:
                    print(f"Warning: {ref} not found in AVrC_allsequences.fasta")
