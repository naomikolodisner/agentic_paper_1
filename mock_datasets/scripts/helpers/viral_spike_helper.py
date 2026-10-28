#!/usr/bin/env python3
"""
Helper for the ART-based viral spike-in step (called by 10_read_mixing.sh).

  log-actual-coverage      After ART has run, compute each genome's actually
                           achieved coverage from its real FASTQ read count
                           (not the -f target) and log it. Coverage is always
                           a derived, logged quantity -- never an input.

The retired percentage design's total-reads / weighted-coverage-plan
subcommands were removed 2026-10-27 (see git history).
"""

import argparse
import sys
from pathlib import Path

from Bio import SeqIO


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
