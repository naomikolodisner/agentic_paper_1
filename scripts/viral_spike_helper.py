#!/usr/bin/env python3
"""
Helper for the ART-based viral spike-in step. Three subcommands:

  total-reads             Compute the total viral read target (R1+R2
                           combined, matching ISS's own --n_reads convention)
                           so that viral reads make up an exact percentage of
                           the final mixed sample.

  weighted-coverage-plan   Given that total, split it UNEVENLY across a
                           sample's genome subset by cycling through
                           config.ART_RATIO_WEIGHTS (the old
                           equal2/equal3/equal4/unequal2/unequal3 scripts'
                           fixed coverage values, reused here as relative
                           weights instead of absolute coverages), then
                           convert each genome's read share into the ART -f
                           fold-coverage value needed to hit it, using that
                           genome's own real length.

  log-actual-coverage      After ART has run, compute each genome's actually
                           achieved coverage from its real FASTQ read count
                           (not the -f target) and log it. Coverage is always
                           a derived, logged quantity -- never an input.
"""

import argparse
import sys
from pathlib import Path

from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def cmd_total_reads(args):
    p = args.pct
    if not (0 < p < 1):
        print(f"ERROR: --pct must be between 0 and 1 (got {p})", file=sys.stderr)
        return 1
    total = round(p * args.background_pairs * 2 / (1 - p))
    print(total)
    return 0


def cmd_weighted_coverage_plan(args):
    genome_lengths = {
        record.id: len(record.seq)
        for record in SeqIO.parse(str(args.contigs), "fasta")
    }
    genome_ids = list(genome_lengths.keys())
    if not genome_ids:
        print(f"ERROR: no genomes found in {args.contigs}", file=sys.stderr)
        return 1

    weights = config.ART_RATIO_WEIGHTS
    assigned_weights = [weights[i % len(weights)] for i in range(len(genome_ids))]
    total_weight = sum(assigned_weights)

    rows = []
    for genome_id, weight in zip(genome_ids, assigned_weights):
        genome_length = genome_lengths[genome_id]
        n_reads_genome = args.total_reads * weight / total_weight
        n_pairs_genome = n_reads_genome / 2
        coverage = (n_pairs_genome * 2 * args.read_length) / genome_length
        rows.append((genome_id, genome_length, weight, n_pairs_genome, coverage))

    with open(args.out, "w") as out_f:
        out_f.write("genome_id\tgenome_length\tweight\tn_pairs_target\tcoverage_target\n")
        for genome_id, genome_length, weight, n_pairs_genome, coverage in rows:
            out_f.write(
                f"{genome_id}\t{genome_length}\t{weight}\t"
                f"{n_pairs_genome:.1f}\t{coverage:.4f}\n"
            )

    print(f"Wrote {len(rows)} genome coverage-plan rows to {args.out}")
    return 0


def cmd_log_actual_coverage(args):
    genome_lengths = {
        record.id: len(record.seq)
        for record in SeqIO.parse(str(args.contigs), "fasta")
    }

    rows = []
    with open(args.actual_pairs_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            genome_id, n_pairs_str = line.split("\t")
            n_pairs_actual = float(n_pairs_str)
            genome_length = genome_lengths.get(genome_id)
            if genome_length is None:
                print(
                    f"WARNING: {genome_id} in {args.actual_pairs_file} not found in "
                    f"{args.contigs} -- skipping coverage calc",
                    file=sys.stderr,
                )
                continue
            coverage = (n_pairs_actual * 2 * args.read_length) / genome_length
            rows.append((genome_id, n_pairs_actual, genome_length, coverage))

    with open(args.out, "w") as out_f:
        out_f.write("genome_id\tn_pairs_actual\tgenome_length\tcoverage_x\n")
        for genome_id, n_pairs_actual, genome_length, coverage in rows:
            out_f.write(
                f"{genome_id}\t{n_pairs_actual:.1f}\t{genome_length}\t{coverage:.4f}\n"
            )

    print(f"Wrote {len(rows)} genome coverage rows to {args.out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_total = sub.add_parser(
        "total-reads",
        help="Compute the total viral read target for a spike-in percentage.",
    )
    p_total.add_argument(
        "--background-pairs", type=int, required=True,
        help="Actual background read-PAIR count (n_pairs_actual from the background manifest).",
    )
    p_total.add_argument(
        "--pct", type=float, required=True,
        help="Target viral read fraction of the final mixed sample (e.g. 0.01 for 1%%).",
    )
    p_total.set_defaults(func=cmd_total_reads)

    p_plan = sub.add_parser(
        "weighted-coverage-plan",
        help="Split the total viral read budget unevenly across a genome subset via ART_RATIO_WEIGHTS.",
    )
    p_plan.add_argument("--contigs", type=Path, required=True)
    p_plan.add_argument(
        "--total-reads", type=float, required=True,
        help="Total viral reads (both mates) to split across the subset.",
    )
    p_plan.add_argument("--read-length", type=int, required=True)
    p_plan.add_argument("--out", type=Path, required=True)
    p_plan.set_defaults(func=cmd_weighted_coverage_plan)

    p_log = sub.add_parser(
        "log-actual-coverage",
        help="Compute derived per-genome coverage from ART's real per-genome read counts.",
    )
    p_log.add_argument("--contigs", type=Path, required=True)
    p_log.add_argument(
        "--actual-pairs-file", type=Path, required=True,
        help="TSV of genome_id<TAB>n_pairs_actual, one row per genome.",
    )
    p_log.add_argument("--read-length", type=int, required=True)
    p_log.add_argument("--out", type=Path, required=True)
    p_log.set_defaults(func=cmd_log_actual_coverage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
