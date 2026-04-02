import os
import gzip
import logging
from Bio import SeqIO

# directories
genomad_dir = "/xdisk/gwatts/kolodisner/fmt_viruses/viral_detection_pipeline/results/03A_genomad"
spades_dir = "/xdisk/gwatts/kolodisner/fmt_viruses/out_spades"
output_base = "/xdisk/gwatts/kolodisner/agentic_paper_1/dataset_creation/no_virus_contigs"

os.makedirs(output_base, exist_ok=True)

# setup logging
log_file = os.path.join(output_base, "removal_log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def open_fasta(file_path):
    """Open fasta whether gzipped or not."""
    if file_path.endswith(".gz"):
        return gzip.open(file_path, "rt")
    return open(file_path, "r")

# loop over samples
for sample in os.listdir(spades_dir):
    spades_sample_dir = os.path.join(spades_dir, sample)

    if not os.path.isdir(spades_sample_dir):
        continue

    # handle both .fasta and .fasta.gz
    contigs_fasta = os.path.join(spades_sample_dir, "contigs.fasta")
    contigs_fasta_gz = contigs_fasta + ".gz"

    if os.path.exists(contigs_fasta):
        contigs_file = contigs_fasta
    elif os.path.exists(contigs_fasta_gz):
        contigs_file = contigs_fasta_gz
    else:
        logging.warning(f"{sample}: no contigs file found")
        continue

    viral_fasta = os.path.join(
        genomad_dir, sample, "contigs_summary", "contigs_virus.fna"
    )

    if not os.path.exists(viral_fasta):
        logging.warning(f"{sample}: no viral file found")
        continue

    logging.info(f"Processing {sample}")

    # get viral IDs
    with open_fasta(viral_fasta) as vf:
        viral_ids = set(record.id for record in SeqIO.parse(vf, "fasta"))

    logging.info(f"{sample}: {len(viral_ids)} viral contigs")

    output_file = os.path.join(output_base, f"{sample}_no_virus.fasta")

    kept = 0
    removed = 0

    with open(output_file, "w") as out_f:
        with open_fasta(contigs_file) as cf:
            for record in SeqIO.parse(cf, "fasta"):
                if record.id not in viral_ids:
                    SeqIO.write(record, out_f, "fasta")
                    kept += 1
                else:
                    removed += 1

    logging.info(f"{sample}: kept={kept}, removed={removed}")

logging.info("Processing complete.")
