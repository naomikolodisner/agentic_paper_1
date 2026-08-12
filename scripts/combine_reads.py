#!/usr/bin/env python3
"""
Concatenate two paired FASTQ read sets (viral spike-in + background) and
shuffle them together into a single mixed sample.

Replaces the old gen_titration_sample.py now that background reads come from
InSilicoSeq (scripts/generate_background.sh) and viral spike-in reads from
ART Illumina (scripts/read_mixing.sh), both simulated at the same read length
(read_mixing.sh forces ART's -l to match the chosen background row's measured
length) with exact, pre-computed read counts on both sides, so there is no
more depth math or seqtk subsampling to do here -- this is a pure
shuffle-merge.
"""

import sys
import gzip
import logging
import argparse
import subprocess
from pathlib import Path

from Bio import SeqIO

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SHUFFLE_SHELL = """
export TMPDIR="{tmpdir}"

paste  <(cat {r1} {b1}) \\
       <(cat {r2} {b2}) \\
    | paste - - - - \\
    | shuf \\
    | awk -F'\\t' '{{OFS="\\n"; print $1,$3,$5,$7 > "{out_prefix}.1.fq";\\
        print $2,$4,$6,$8 > "{out_prefix}.2.fq"}}'

gzip "{out_prefix}.1.fq"
gzip "{out_prefix}.2.fq"
"""


def open_compressed(file):
    if str(file).endswith(".gz"):
        return gzip.open(file, "rt")
    return open(file, "r")


def first_read_length(fastq_path):
    with open_compressed(fastq_path) as f:
        return len(next(SeqIO.parse(f, "fastq")).seq)


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate and shuffle a viral spike-in read set with a background read set."
    )
    parser.add_argument('-R1', type=Path, required=True, help="Viral spike-in FastQ reads file 1")
    parser.add_argument('-R2', type=Path, required=True, help="Viral spike-in FastQ reads file 2")
    parser.add_argument('-B1', type=Path, required=True, help="Background FastQ reads file 1")
    parser.add_argument('-B2', type=Path, required=True, help="Background FastQ reads file 2")
    parser.add_argument('-s', '--seed', type=int, default=None, required=False,
                         help="Seed used upstream to generate R1/R2 and B1/B2 (logged only).")
    parser.add_argument('-o', '--output-prefix', required=True, help="Output filename prefix")

    args = parser.parse_args()

    if args.seed is not None:
        logger.info("Upstream seed: %d", args.seed)

    viral_len = first_read_length(args.R1)
    background_len = first_read_length(args.B1)
    if viral_len != background_len:
        logger.error(
            "Read length of viral spike-in and background reads differ (%d != %d) "
            "-- were they generated with the same ISS --model?",
            viral_len, background_len,
        )
        return 1

    if args.R1.name.endswith('.gz'):
        r1 = f'<(zcat "{args.R1}")'
        r2 = f'<(zcat "{args.R2}")'
    else:
        r1 = f'"{args.R1}"'
        r2 = f'"{args.R2}"'

    if args.B1.name.endswith('.gz'):
        b1 = f'<(zcat "{args.B1}")'
        b2 = f'<(zcat "{args.B2}")'
    else:
        b1 = f'"{args.B1}"'
        b2 = f'"{args.B2}"'

    logger.info("Shuffling reads and gzipping output files...")
    result = subprocess.run(
        SHUFFLE_SHELL.format(
            b1=b1, b2=b2, r1=r1, r2=r2,
            out_prefix=args.output_prefix,
            tmpdir="/tmp",
        ),
        shell=True, executable='/bin/bash',
    )
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
