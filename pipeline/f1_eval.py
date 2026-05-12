"""
F1, precision, and recall evaluation for viral detection results.

Ground truth comes from contig_identities.csv (superkingdom == "Viruses").
Predicted contigs come from a tool's output FASTA file.

Profile/model filtering is required for accurate recall. The same contig ID
(e.g. k141_0) appears in every assembly because SPAdes reuses sequential
numbering. Without filtering, the FN denominator includes viral contigs from
all 15 simulated communities — but the tool only ran on one, so it can never
find the others. Filter to the specific profile+model so the denominator
contains only the viral contigs that actually existed in the assembly the tool
analyzed.

The sample directory name encodes both fields: SRR4831655_hiseq
  → profile = "SRR4831655", model = "hiseq"
"""

import csv
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #

F1_GOOD     = 0.70   # >= this is a "good" result
F1_TERRIBLE = 0.50   # <  this gets dropped from exploration


# --------------------------------------------------------------------------- #
# Data class
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    tool: str
    sample_id: str
    coverage: str
    sample_type: str
    profile: str
    model: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def is_good(self) -> bool:
        return self.f1 >= F1_GOOD

    def is_terrible(self) -> bool:
        return self.f1 < F1_TERRIBLE

    def status(self) -> str:
        if self.is_good():
            return "GOOD"
        if self.is_terrible():
            return "TERRIBLE"
        return "ACCEPTABLE"

    def summary(self) -> str:
        return (
            f"Tool={self.tool}  Sample={self.sample_id}  "
            f"Coverage={self.coverage or 'N/A'}  Type={self.sample_type or 'N/A'}\n"
            f"  TP={self.tp}  FP={self.fp}  FN={self.fn}\n"
            f"  Precision={self.precision:.4f}  Recall={self.recall:.4f}  "
            f"F1={self.f1:.4f}  [{self.status()}]"
        )


# --------------------------------------------------------------------------- #
# Ground truth loading
# --------------------------------------------------------------------------- #

def load_viral_contig_ids(
    ground_truth_csv: Path = config.CONTIG_IDENTITIES,
    profile: Optional[str] = None,
    model: Optional[str] = None,
) -> set:
    """
    Return the set of contig IDs whose superkingdom is 'Viruses'.

    Always pass profile and model (e.g. 'SRR4831655', 'hiseq') so that only
    the contigs belonging to that specific assembly are in the denominator.
    The CSV is ~1.1 M rows; load once and reuse the returned set across calls.
    """
    viral_ids = set()
    with open(ground_truth_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if profile and row["profile"] != profile:
                continue
            if model and row["model"] != model:
                continue
            if row["superkingdom"] == "Viruses":
                viral_ids.add(row["query_id"])
    return viral_ids


# --------------------------------------------------------------------------- #
# FASTA parsing
# --------------------------------------------------------------------------- #

def parse_fasta_ids(fasta_path: Path) -> set:
    """
    Extract contig IDs from a viral detection tool's output FASTA.

    Handles both common header styles:
      >k141_2191, av_score: 0.605        (Seeker)
      >k141_18325 flag=1 multi=4 len=562 (SPAdes-style)
    """
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                token = line[1:].strip().split(",")[0].split()[0]
                ids.add(token)
    return ids


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def compute_metrics(predicted: set, ground_truth_viral: set) -> tuple:
    """Return (precision, recall, f1, tp, fp, fn)."""
    tp = len(predicted & ground_truth_viral)
    fp = len(predicted - ground_truth_viral)
    fn = len(ground_truth_viral - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1, tp, fp, fn


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def evaluate(
    fasta_path: Path,
    ground_truth_viral: set,
    tool: str,
    sample_id: str,
    coverage: str = "",
    sample_type: str = "",
    profile: str = "",
    model: str = "",
) -> EvalResult:
    """
    Evaluate one tool FASTA against a pre-loaded ground truth set.

    Prefer pre-loading ground_truth_viral with load_viral_contig_ids() and
    reusing it across calls rather than reloading the CSV each time.
    """
    predicted = parse_fasta_ids(fasta_path)
    precision, recall, f1, tp, fp, fn = compute_metrics(predicted, ground_truth_viral)
    return EvalResult(
        tool=tool, sample_id=sample_id,
        coverage=coverage, sample_type=sample_type,
        profile=profile, model=model,
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
    )


def evaluate_batch(
    samples: list[dict],
    ground_truth_csv: Path = config.CONTIG_IDENTITIES,
) -> list[EvalResult]:
    """
    Evaluate multiple samples, reading the ground truth CSV once per unique
    (profile, model) pair.

    Each dict must contain: fasta_path, tool, sample_id
    Optionally: profile, model, coverage, sample_type
    """
    cache: dict[tuple, set] = {}
    results = []
    for s in samples:
        key = (s.get("profile"), s.get("model"))
        if key not in cache:
            cache[key] = load_viral_contig_ids(ground_truth_csv, *key)
        results.append(evaluate(
            fasta_path=Path(s["fasta_path"]),
            ground_truth_viral=cache[key],
            tool=s["tool"],
            sample_id=s["sample_id"],
            coverage=s.get("coverage", ""),
            sample_type=s.get("sample_type", ""),
            profile=s.get("profile", ""),
            model=s.get("model", ""),
        ))
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate viral detection precision, recall, and F1."
    )
    parser.add_argument("fasta", help="FASTA file of predicted viral contigs")
    parser.add_argument(
        "--ground-truth",
        default=str(config.CONTIG_IDENTITIES),
        help="Path to contig_identities.csv",
    )
    parser.add_argument("--tool",        required=True)
    parser.add_argument("--sample-id",   required=True)
    parser.add_argument("--profile",     default=None, help="e.g. SRR4831655")
    parser.add_argument("--model",       default=None, help="e.g. hiseq | miseq | novaseq")
    parser.add_argument("--coverage",    default="")
    parser.add_argument("--sample-type", default="")
    args = parser.parse_args()

    viral_ids = load_viral_contig_ids(Path(args.ground_truth), args.profile, args.model)
    result = evaluate(
        fasta_path=Path(args.fasta),
        ground_truth_viral=viral_ids,
        tool=args.tool,
        sample_id=args.sample_id,
        coverage=args.coverage,
        sample_type=args.sample_type,
        profile=args.profile or "",
        model=args.model or "",
    )
    print(result.summary())
