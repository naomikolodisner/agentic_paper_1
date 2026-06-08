import asyncio
import parsl
from parsl import python_app
from academy.agent import Agent, action

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# Parsl app
# --------------------------------------------------------------------------- #

@python_app(executors=['checkv_htex'])
def checkv_app(
    checkv_parser, parse_length, work_dir,
    unzipped_spades, viral_result, checkv_output_dir,
    parse_input, selection_csv, checkvdb,
):
    import os
    import subprocess
    import socket
    import shutil
    print("CheckV Running on node:", socket.gethostname(), flush=True)

    # Guard: if the upstream tool found no viral sequences, the input FASTA will
    # be absent (or empty).  Skip CheckV rather than crashing.
    if not os.path.isfile(viral_result) or os.path.getsize(viral_result) == 0:
        print(
            f"[CheckV] Input file missing or empty — no viral sequences detected. "
            f"Skipping CheckV for: {viral_result}",
            flush=True,
        )
        return None, 0.0

    if os.path.exists(checkv_output_dir):
        shutil.rmtree(checkv_output_dir)
    os.makedirs(checkv_output_dir)

    cmd_checkv = [
        "conda", "run", "-n", "checkv", "checkv", "end_to_end",
        viral_result, checkv_output_dir, "-t", "4", "-d", checkvdb,
    ]
    cmd_parser = [
        "conda", "run", "-n", "r", "Rscript", checkv_parser,
        "-i", parse_input, "-l", str(parse_length), "-o", selection_csv,
    ]
    subprocess.run(cmd_checkv, check=True)
    subprocess.run(cmd_parser, check=True)

    cleaned_selection_csv = selection_csv.replace(".csv", "_cleaned.csv")
    with open(selection_csv, "r") as infile, open(cleaned_selection_csv, "w") as outfile:
        for line in infile:
            if line.startswith("contig_id"):
                continue  # seqtk subseq requires a plain ID list; skip the header row
            else:
                clean_line = line.split("||")[0].strip()
                outfile.write(f"{clean_line}\n")

    cmd_seqtk = [
        "conda", "run", "-n", "seqtk", "seqtk", "subseq",
        unzipped_spades, cleaned_selection_csv,
    ]
    subset_spades = os.path.join(checkv_output_dir, "subset_spades.fasta")
    with open(subset_spades, "w") as out_f:
        subprocess.run(cmd_seqtk, check=True, stdout=out_f)

    if os.path.getsize(subset_spades) == 0:
        print("[CheckV] subset_spades.fasta is empty — no sequences passed quality filter.", flush=True)
        return None, 0.0

    # quality_ratio: fraction of detected contigs rated High-quality by CheckV.
    # Kept as a future-use evaluation signal alongside F1.
    total = 0
    high_quality = 0
    quality_tsv = os.path.join(checkv_output_dir, "quality_summary.tsv")
    with open(quality_tsv, "r") as f:
        for line in f:
            if line.startswith("contig_id") or not line.strip():
                continue
            columns = line.strip().split("\t")
            if len(columns) < 8:
                continue
            total += 1
            if columns[7].strip() == "High-quality":
                high_quality += 1
    quality_ratio = high_quality / total if total > 0 else 0.0

    return subset_spades, quality_ratio


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class CheckVAgent(Agent):
    def __init__(self):
        super().__init__()

    @action
    async def run_checkv(
        self, checkv_parser, parse_length, work_dir,
        unzipped_spades, viral_result, checkv_output_dir,
        parse_input, selection_csv, checkvdb,
    ):
        future = checkv_app(
            checkv_parser, parse_length, work_dir,
            unzipped_spades, viral_result, checkv_output_dir,
            parse_input, selection_csv, checkvdb,
        )
        subset_spades, quality_ratio = await asyncio.to_thread(future.result)
        return subset_spades, quality_ratio
