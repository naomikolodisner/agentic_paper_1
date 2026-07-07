#!/usr/bin/env python3
"""
Helper for the ISS-based viral spike-in step. Two subcommands:

  total-reads   Compute the --n_reads target for ISS's viral --genomes call so
                that viral reads make up an exact percentage of the final
                mixed sample.

  coverage-log  After ISS has simulated the viral spike (using its own
                --abundance distribution to split the total across the
                subset's genomes), compute each genome's derived coverage
                from ISS's own *_abundance.txt output + genome length + the
                model's actual read length, and write coverage_log.txt.

Coverage is always a derived, logged quantity here -- never an input.
"""

import argparse
import sys
from pathlib import Path

from Bio import SeqIO


def cmd_total_reads(args):
    p = args.pct
    if not (0 < p < 1):
        print(f"ERROR: --pct must be between 0 and 1 (got {p})", file=sys.stderr)
        return 1
    total = round(p * args.background_pairs * 2 / (1 - p))
    print(total)
    return 0


def cmd_coverage_log(args):
    genome_lengths = {
        record.id: len(record.seq)
        for record in SeqIO.parse(str(args.contigs), "fasta")
    }

    rows = []
    with open(args.abundance_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            genome_id, proportion_str = line.split("\t")
            proportion = float(proportion_str)
            genome_length = genome_lengths.get(genome_id)
            if genome_length is None:
                print(
                    f"WARNING: {genome_id} in {args.abundance_file} not found in "
                    f"{args.contigs} -- skipping coverage calc",
                    file=sys.stderr,
                )
                continue
            n_reads_genome = proportion * args.total_reads
            coverage = (n_reads_genome * args.read_length) / genome_length
            rows.append((genome_id, proportion, n_reads_genome, genome_length, coverage))

    with open(args.out, "w") as out_f:
        out_f.write("genome_id\tabundance_proportion\tn_reads\tgenome_length\tcoverage_x\n")
        for genome_id, proportion, n_reads_genome, genome_length, coverage in rows:
            out_f.write(
                f"{genome_id}\t{proportion:.6f}\t{n_reads_genome:.1f}\t"
                f"{genome_length}\t{coverage:.4f}\n"
            )

    print(f"Wrote {len(rows)} genome coverage rows to {args.out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_total = sub.add_parser(
        "total-reads",
        help="Compute the ISS --n_reads target for a viral spike-in percentage.",
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

    p_cov = sub.add_parser(
        "coverage-log",
        help="Compute derived per-genome coverage from ISS's own abundance output.",
    )
    p_cov.add_argument("--abundance-file", type=Path, required=True)
    p_cov.add_argument("--contigs", type=Path, required=True)
    p_cov.add_argument("--read-length", type=int, required=True)
    p_cov.add_argument(
        "--total-reads", type=float, required=True,
        help="Total viral reads (both mates) ISS was asked to generate for this spike.",
    )
    p_cov.add_argument("--out", type=Path, required=True)
    p_cov.set_defaults(func=cmd_coverage_log)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
