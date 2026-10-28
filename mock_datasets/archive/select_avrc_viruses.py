#!/usr/bin/env python3
"""
Select AVrC viruses for spike-in, fully using the AVrC toolkit programmatically.
"""

import random
from pathlib import Path
import pandas as pd
from click.testing import CliRunner
from avrc.commands.filter import filter_cmd

# CONFIG
AVRC_FASTA_DIR = Path("/path/to/avrc_fastas")  # where AVrC FASTAs live
METADATA_CSV = AVRC_FASTA_DIR / "AvRCv1.Merged_ViralDesc.csv"
OUTPUT_DIR = Path("avrc_samples")
NUM_SAMPLES = 20
NUM_VIRUSES = [1, 2, 3, 4]
TAXONOMY_MODE = ["all", "same_family", "same_class"]
SEED = 42

random.seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

# LOAD METADATA
df = pd.read_csv(METADATA_CSV)
df = df[df["Family"] != "Unclassified"]  # optional filter

# Group contigs by vOTU
votu_groups = df.groupby("vOTU_ID")

# HELPER: filter AVrC sequences programmatically
def filter_avrc_sequences(input_path: Path, quality="High-quality", output_type="both"):
    """Call AVrC filter_cmd programmatically."""
    runner = CliRunner()
    result = runner.invoke(
        filter_cmd,
        [str(input_path), "--quality", quality, "--output", output_type]
    )
    if result.exit_code != 0:
        raise RuntimeError(f"AVrC filter failed on {input_path}: {result.output}")
    return result.output 

# MAIN SELECTION LOOP
for i in range(NUM_SAMPLES):
    num = random.choice(NUM_VIRUSES)
    mode = random.choice(TAXONOMY_MODE)

    # Taxonomy-based subsetting
    if mode == "same_family":
        fam = random.choice(df["Family"].unique())
        subset = df[df["Family"] == fam]
    elif mode == "same_class":
        cls = random.choice(df["Class"].unique())
        subset = df[df["Class"] == cls]
    else:
        subset = df

    chosen_votus = random.sample(list(subset["vOTU_ID"].unique()), num)

    sample_dir = OUTPUT_DIR / f"sample_{i+1}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    refs_file = sample_dir / "refs.txt"
    with open(refs_file, "w") as f:
        for votu in chosen_votus:
            contigs = votu_groups.get_group(votu)["contig_id"].tolist()
            contig = random.choice(contigs)

            # Apply AVrC filter to the contig FASTA
            fasta_path = AVRC_FASTA_DIR / f"{contig}.fasta"
            filter_output = filter_avrc_sequences(fasta_path)

            # Write contig path to refs.txt
            print(contig, fasta_path, sep="\t", file=f)

print("AVrC virus selection complete. refs.txt files are ready for spike-in.")
