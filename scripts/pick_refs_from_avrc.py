import random
from pathlib import Path
import pandas as pd
from click.testing import CliRunner
from avrc.commands.filter import filter_cmd

# PARAMETERS
AVRC_FASTA_DIR = Path("path to avrc fasta files")  
OUTPUT_DIR = Path("avrc_samples")
NUM_SAMPLES = 20
NUM_VIRUSES = [1, 2, 3, 4]
TAXONOMY_MODE = ["all", "same_family", "same_class"]
SEED = 42

random.seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

# LOAD METADATA
metadata_csv = AVRC_FASTA_DIR / "AvRCv1.Merged_ViralDesc.csv"
df = pd.read_csv(metadata_csv)

# Exclude unclassified viruses
df = df[df["Family"] != "Unclassified"]

# Group by vOTU for random selection
votu_groups = df.groupby("vOTU_ID")

# FUNCTION TO FILTER USING AVrC TOOLKIT
def filter_avrc_sequences(input_dir, quality="High-quality", output_type="both"):
    runner = CliRunner()
    result = runner.invoke(
        filter_cmd,
        [
            str(input_dir),
            "--quality", quality,
            "--output", output_type
        ]
    )
    if result.exit_code != 0:
        raise RuntimeError(f"AVrC filter failed: {result.output}")
    return result.output  

# CREATE SAMPLES
for i in range(NUM_SAMPLES):
    num = random.choice(NUM_VIRUSES)
    mode = random.choice(TAXONOMY_MODE)

    # Subset by taxonomy if required
    if mode == "same_family":
        fam = random.choice(df["Family"].unique())
        subset = df[df["Family"] == fam]
    elif mode == "same_class":
        cls = random.choice(df["Class"].unique())
        subset = df[df["Class"] == cls]
    else:
        subset = df

    chosen_votus = random.sample(list(subset["vOTU_ID"].unique()), num)

    out_dir = OUTPUT_DIR / f"sample_{i+1}"
    out_dir.mkdir(parents=True, exist_ok=True)

    refs_file = out_dir / "refs.txt"
    with open(refs_file, "w") as f:
        for votu in chosen_votus:
            contigs = votu_groups.get_group(votu)["contig_id"].tolist()
            # pick one contig per vOTU (you can extend to multiple if desired)
            contig = random.choice(contigs)

            # Use AVrC filter to ensure sequence meets criteria
            filtered_output = filter_avrc_sequences(AVRC_FASTA_DIR / f"{contig}.fasta")
            # Write filtered FASTA path to refs.txt
            print(contig, f"{AVRC_FASTA_DIR}/{contig}.fasta", sep="\t", file=f)

print("AVrC reference selection complete. Check refs.txt in each sample folder.")
