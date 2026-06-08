#!/usr/bin/env python3
"""
Mix reads from one sample randomly with reads from another sample, at a given
sequencing depth and relative abundance.
"""

import sys
import random
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
import os
from Bio import SeqIO

SEQTK_BIN = "/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/seqtk"

#from strainge_benchmarks import open_compressed
import gzip

def open_compressed(file):
    if str(file).endswith(".gz"):
        return gzip.open(file, "rt")
    return open(file, "r")

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SHUFFLE_SHELL = """
export MEMORY=40
export TMPDIR="{tmpdir}"

paste  <(cat {r1} "{b1}") \\
       <(cat {r2} "{b2}") \\
    | paste - - - - \\
    | shuf \\
    | awk -F'\\t' '{{OFS="\\n"; print $1,$3,$5,$7 > "{out_prefix}.1.fq";\\
        print $2,$4,$6,$8 > "{out_prefix}.2.fq"}}'

gzip "{out_prefix}.1.fq"
gzip "{out_prefix}.2.fq"
"""

def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial titration (spike-in) samples."
    )

    parser.add_argument(
        '-R1', type=Path,
        help="FastQ reads file 1"
    )

    parser.add_argument(
        '-R2', type=Path,
        help="FastQ reads file 2"
    )

    parser.add_argument(
        '-B1', type=Path,
        help="Background sample FASTQ reads file 1"
    )

    parser.add_argument(
        '-B2', type=Path,
        help="Background sample FASTQ reads file 2"
    )

    parser.add_argument(
        '-d', '--depth', type=int,
        help="Depth of sequencing"
    )

    parser.add_argument(
        '-s', '--seed', type=int, default=None, required=False,
        help="Random number generator seed."
    )

    parser.add_argument(
        '-o', '--output-prefix',
        help="Output filename prefix"
    )

    args = parser.parse_args()

    seed = args.seed if args.seed else random.randrange(1, 2**32 - 1)
    logger.info("Random seed: %d", seed)

    # Determine read length for both samples
    in_sample_read_len = None
    with open_compressed(args.R1) as f:
        read = next(SeqIO.parse(f, "fastq"))
        in_sample_read_len = len(read.seq)

    base_sample_read_len = None
    with open_compressed(args.B1) as f:
        read = next(SeqIO.parse(f, "fastq"))
        base_sample_read_len = len(read.seq)

    if in_sample_read_len != base_sample_read_len:
        logger.error("Read length of input sample and base sample is not the "
                     "same (%d != %d)", in_sample_read_len,
                     base_sample_read_len)

        return 1

    # Determine number of reads to sample
    total_read_pairs = int(args.depth / (in_sample_read_len * 2))

    logger.info("Counting number of reads in input sample...")
    with open_compressed(args.R1) as f:
        #p = subprocess.run(f'grep "^@" | wc -l',
        #                   stdin=f, stdout=subprocess.PIPE,
        #                   shell=True, executable='/bin/bash')
        p = subprocess.run(['awk', 'NR%4==1{c++} END{print c}', str(args.R1)], stdout=subprocess.PIPE)

    #num_reads_in = int(p.stdout)
    #reads_from_bg = total_read_pairs - num_reads_in

    num_reads_in = int(p.stdout)
    reads_from_bg = max(0, total_read_pairs - num_reads_in)
    
    logger.info("Total number of read pairs required: %d", total_read_pairs)
    logger.info("Number of read pairs in input sample: %d", num_reads_in)
    logger.info("Number of read pairs from background sample: %d",
                reads_from_bg)

    logger.info("Subsampling reads from background sample...")
    b1 = tempfile.NamedTemporaryFile(mode='w')
    b2 = tempfile.NamedTemporaryFile(mode='w')
    subprocess.run([SEQTK_BIN, "sample", "-s", str(seed), str(args.B1), str(reads_from_bg)],
                   stdout=b1, check=True)
    subprocess.run([SEQTK_BIN, "sample", "-s", str(seed), str(args.B2), str(reads_from_bg)],
                   stdout=b2, check=True)
    b1.flush()
    b2.flush()

    b1_size = os.path.getsize(b1.name)
    b2_size = os.path.getsize(b2.name)
    logger.info("Background sample sizes: B1=%d bytes, B2=%d bytes", b1_size, b2_size)
    if b2_size == 0:
        logger.error("B2 temp file is empty after seqtk — background2.fq may be empty or missing")
        return 1

    b1_count = int(subprocess.run(['awk', 'NR%4==1{c++} END{print c+0}', b1.name], stdout=subprocess.PIPE).stdout)
    b2_count = int(subprocess.run(['awk', 'NR%4==1{c++} END{print c+0}', b2.name], stdout=subprocess.PIPE).stdout)
    logger.info("Background read counts: B1=%d reads, B2=%d reads", b1_count, b2_count)
    if b1_count != b2_count:
        logger.error("B1 and B2 read counts differ (%d vs %d) — background FASTQ has unequal R1/R2 read counts", b1_count, b2_count)
        return 1

    if args.R1.name.endswith('.gz'):
        r1 = f'<(zcat "{args.R1}")'
        r2 = f'<(zcat "{args.R2}")'
    else:
        r1 = f'"{args.R1}"'
        r2 = f'"{args.R2}"'

    logger.info("Shuffling reads and gzipping output files...")
    subprocess.run(
    SHUFFLE_SHELL.format(b1=b1.name, b2=b2.name, r1=r1, r2=r2,
                         out_prefix=args.output_prefix,
                         tmpdir=os.environ.get('TMPDIR', '/tmp')),
    shell=True, executable='/bin/bash'
    )
    b1.close()
    b2.close()


if __name__ == '__main__':
    r = main()
    sys.exit(r if r else 0)
