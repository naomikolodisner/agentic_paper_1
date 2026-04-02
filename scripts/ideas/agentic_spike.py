#!/usr/bin/env python3
"""
Agentic spike-in generator that adaptively changes parameters
(coverage, number of viruses, abundance) and virus selection
to maximize detection F1 score.
"""

import random
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy

# CONFIGURABLE PATHS
GEN_TITRATION_SCRIPT = Path("/path/to/gen_titration_sample.py")
AVRC_DB_DIR = Path("/path/to/AVrC_database")
DETECTION_PIPELINE = Path("/path/to/run_genomad_pipeline.sh") 

OUTPUT_DIR = Path("titration_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# HYPERPARAMETERS & POLICY
policy = {
    "coverage": {5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0},
    "num_viruses": {1: 1.0, 5: 1.0, 10: 1.0},
    "abundance": {0.1: 1.0, 1.0: 1.0, 5.0: 1.0}  # Percent of reads
}

ALPHA = 0.1  # Learning rate for policy updates
BETA = 0.05  # Learning rate for virus selection

MAX_ITER = 20  # Number of experiments per run

# LOAD VIRUS REFERENCE METADATA
avrc_csv = AVRC_DB_DIR / "AvRCv1.Merged_ViralDesc.csv"
virus_metadata = pd.read_csv(avrc_csv)
virus_ids = virus_metadata["contig_id"].tolist()

# Initialize virus selection probabilities
virus_probs = {v: 1.0 for v in virus_ids}

# HELPER FUNCTIONS
def weighted_choice(d):
    items = list(d.keys())
    weights = np.array(list(d.values()), dtype=float)
    weights /= weights.sum()
    return np.random.choice(items, p=weights)

def sample_parameters(policy):
    return (
        weighted_choice(policy["coverage"]),
        weighted_choice(policy["num_viruses"]),
        weighted_choice(policy["abundance"])
    )

def sample_viruses(virus_probs, n):
    """Select n viruses based on adaptive probabilities."""
    ids = list(virus_probs.keys())
    probs = np.array(list(virus_probs.values()), dtype=float)
    probs /= probs.sum()
    return list(np.random.choice(ids, size=n, replace=False, p=probs))

def generate_spike_in_sample(coverage, abundance, virus_list, iteration):
    """Run gen_titration_sample.py to create a titration sample."""
    # For simplicity, assume background sample R1/R2 are fixed
    bg_R1 = "background_R1.fastq.gz"
    bg_R2 = "background_R2.fastq.gz"
    # Viral reads FASTQ files need to be generated from AVrC
    # Example: use selected virus_list to create viral_R1/R2
    viral_R1 = f"viral_{iteration}_R1.fastq.gz"
    viral_R2 = f"viral_{iteration}_R2.fastq.gz"
    # TODO: generate viral FASTQs using AVrC toolkit for selected viruses
    # For now assume these files exist
    
    out_prefix = OUTPUT_DIR / f"titration_{iteration}"
    cmd = [
        "python3", str(GEN_TITRATION_SCRIPT),
        "-R1", viral_R1, "-R2", viral_R2,
        "-B1", bg_R1, "-B2", bg_R2,
        "-d", str(coverage),
        "-o", str(out_prefix)
    ]
    subprocess.run(cmd, check=True)
    return out_prefix

def run_detection_pipeline(sample_prefix):
    """Run genomad pipeline and return detected virus IDs."""
    # Example command
    cmd = [str(DETECTION_PIPELINE), str(sample_prefix)]
    subprocess.run(cmd, check=True)
    # Parse pipeline output (e.g., contigs_virus_summary.tsv)
    detected_file = sample_prefix / "contigs_virus_summary.tsv"
    detected = []
    with open(detected_file) as f:
        for line in f:
            if line.startswith("seq_name"):
                continue
            detected.append(line.strip().split("\t")[0])
    return detected

def compute_f1(truth, detected):
    """Compute F1 score."""
    truth_set = set(truth)
    detected_set = set(detected)
    tp = len(truth_set & detected_set)
    fp = len(detected_set - truth_set)
    fn = len(truth_set - detected_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def update_policy(policy, params, f1, baseline):
    """Update parameter selection probabilities based on F1 improvement."""
    coverage, num_viruses, abundance = params
    delta = f1 - baseline
    policy["coverage"][coverage] += ALPHA * delta
    policy["num_viruses"][num_viruses] += ALPHA * delta
    policy["abundance"][abundance] += ALPHA * delta

def update_virus_probs(virus_probs, viruses, f1, baseline):
    """Update virus selection probabilities based on F1 improvement."""
    delta = f1 - baseline
    for v in viruses:
        virus_probs[v] += BETA * delta

# AGENT LOOP
baseline_F1 = 0.0
for iteration in range(1, MAX_ITER + 1):
    # Sample parameters & viruses
    coverage, num_viruses, abundance = sample_parameters(policy)
    selected_viruses = sample_viruses(virus_probs, num_viruses)

    # Generate titration sample
    sample_prefix = generate_spike_in_sample(coverage, abundance, selected_viruses, iteration)

    # Run detection pipeline
    detected = run_detection_pipeline(sample_prefix)

    # Compute F1
    f1 = compute_f1(selected_viruses, detected)
    print(f"[Iteration {iteration}] Params: {coverage},{num_viruses},{abundance}% F1: {f1:.3f}")

    # Update policies
    update_policy(policy, (coverage, num_viruses, abundance), f1, baseline_F1)
    update_virus_probs(virus_probs, selected_viruses, f1, baseline_F1)

    # Update baseline
    baseline_F1 = 0.9 * baseline_F1 + 0.1 * f1  # moving average
