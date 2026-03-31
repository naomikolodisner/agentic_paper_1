import pandas as pd
from Bio import SeqIO

viral_summary = pd.read_csv("contigs_virus_summary.tsv", sep="\t")

# keep only high-confidence viral contigs
viral_ids = set(
    viral_summary.loc[viral_summary["virus_score"] > 0.95, "seq_name"]
)

with open("contigs_no_virus.fasta", "w") as out_f:
    for record in SeqIO.parse("contigs.fasta", "fasta"):
        if record.id not in viral_ids:
            SeqIO.write(record, out_f, "fasta")
