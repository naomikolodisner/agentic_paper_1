# select_avrc_viruses_adaptive.py
"""
Select AVrC viruses adaptively based on F1 feedback from spike-in experiments.
Uses the AVrC toolkit filter command and updates selection probabilities.
"""

import pandas as pd
import random
from pathlib import Path
from click.testing import CliRunner
from avrc.commands.filter import filter_cmd

# Paths
avrc_dir = Path("AVrC_database")  # Directory with AVrC fasta + metadata
fasta_dir = Path("metagenomes_no_viruses")  # Your cleaned metagenomes
refs_file = Path("refs.txt")  # Output selected viruses

# Features and F1 scores
features = ["pred_lifestyle", "Realm", "Kingdom"]
f1_scores = {feat: {} for feat in features}  # Tracks F1 per feature value

# Function to mock F1 feedback
def get_f1_for_virus(virus_id):
    """
    Replace with actual evaluation after spiking in viruses.
    Returns F1 score for the given virus.
    """
    return random.uniform(0.4, 0.95)

# Run AVrC filter command
def filter_avrc(quality="High-quality", additional_args=None):
    """
    Calls AVrC toolkit's filter command to select viruses matching criteria.
    Returns pandas DataFrame of filtered viruses.
    """
    runner = CliRunner()
    args = [str(avrc_dir), "--quality", quality, "--output", "metadata"]
    if additional_args:
        args.extend(additional_args)

    result = runner.invoke(filter_cmd, args)

    if result.exit_code != 0:
        raise RuntimeError(f"AVrC filter failed:\n{result.output}")

    # Parse output metadata into DataFrame
    # AVrC metadata CSV is written in the output folder
    # For simplicity, we assume it writes 'filtered_metadata.csv'
    metadata_csv = avrc_dir / "filtered_metadata.csv"
    df = pd.read_csv(metadata_csv)
    return df

# Assign selection weights based on F1
def assign_f1_weights(df):
    weights = []
    for _, row in df.iterrows():
        w = 1.0
        for feat in features:
            val = row[feat]
            w *= f1_scores[feat].get(val, 0.5)  # default 0.5 if unseen
        weights.append(w)
    return weights

# Adaptive selection of viruses
def select_adaptive_viruses(n=20, quality="High-quality", extra_args=None):
    df_filtered = filter_avrc(quality=quality, additional_args=extra_args)
    weights = assign_f1_weights(df_filtered)
    selected = df_filtered.sample(n=min(n, len(df_filtered)), weights=weights)
    return selected

# Main execution
if __name__ == "__main__":
    # Step 1: Select viruses adaptively
    selected_df = select_adaptive_viruses(n=20)

    # Step 2: Write selected contig IDs to refs.txt
    selected_df["contig_id"].to_csv(refs_file, index=False, header=False)
    print(f"Saved {len(selected_df)} adaptive AVrC virus IDs to {refs_file}")

    # Step 3: Update F1 scores based on evaluation
    for _, row in selected_df.iterrows():
        f1 = get_f1_for_virus(row["contig_id"])
        for feat in features:
            val = row[feat]
            prev = f1_scores[feat].get(val, 0.5)
            f1_scores[feat][val] = 0.5 * prev + 0.5 * f1
