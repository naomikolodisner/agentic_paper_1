import asyncio
import os
import shutil
import parsl
from parsl import python_app
from academy.agent import Agent, action

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.parsl_configs import blast_config


# --------------------------------------------------------------------------- #
# Parsl apps
# --------------------------------------------------------------------------- #

@python_app
def split_fasta_app(fasta_file, split_dir, split_size):
    import os
    import subprocess
    import shutil
    import socket
    print("Split Fasta Running on node:", socket.gethostname(), flush=True)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)
    cmd = [
        "conda", "run", "-n", "fasplit_env",
        "faSplit", "about", fasta_file, str(split_size), f"{split_dir}/",
    ]
    subprocess.run(cmd, check=True)
    return split_dir


@python_app
def make_blast_db_app(db_dir, max_db_size, db_list_path):
    import os
    import subprocess
    import socket
    print("Make Blast Db Running on node:", socket.gethostname(), flush=True)
    os.makedirs(db_dir, exist_ok=True)
    os.chdir(db_dir)

    with open(db_list_path, "w") as db_list:
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".fasta"):
                    rel_path = os.path.join(root, file).lstrip("./")
                    rel_path_no_ext = os.path.splitext(rel_path)[0]
                    db_list.write(rel_path_no_ext + "\n")

    if not os.path.exists(db_list_path) or os.path.getsize(db_list_path) == 0:
        raise FileNotFoundError(f"Empty or missing db list: {db_list_path}")

    with open(db_list_path) as f:
        for line in f:
            db_file_base = line.strip()
            db_name = os.path.splitext(os.path.basename(db_file_base))[0]
            db_prefix = os.path.join(db_dir, db_name)
            fasta_path = db_file_base + ".fasta"
            if all(os.path.exists(f"{db_prefix}.{ext}") for ext in ["nhr", "nin", "nsq"]):
                continue
            cmd = [
                "conda", "run", "-n", "blast_env",
                "makeblastdb",
                "-title", db_name,
                "-out", db_prefix,
                "-in", fasta_path,
                "-dbtype", "nucl",
                "-max_file_sz", str(max_db_size),
            ]
            subprocess.run(cmd, check=True)
    return "blast_db_complete"


@python_app
def run_blast_app(split_dir, blast_results_dir, db_dir, blast_type, eval_param, out_fmt, max_target_seqs):
    import os
    import subprocess
    import socket
    print("BLAST Running on node:", socket.gethostname(), flush=True)
    split_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".fa"))
    db_list_path = os.path.join(db_dir, "db-list")

    with open(db_list_path, "r") as f:
        databases = [line.strip() for line in f.readlines()]

    for split_file in split_files:
        for db in databases:
            db_base = os.path.splitext(db)[0]
            result_dir = os.path.join(blast_results_dir, db_base, split_file)
            os.makedirs(result_dir, exist_ok=True)
            blast_out = os.path.join(result_dir, f"{split_file}.blastout")
            blast_db = os.path.join(db_dir, db_base)
            cmd = [
                "conda", "run", "-n", "blast_env", blast_type,
                "-num_threads", "48",
                "-db", blast_db,
                "-query", os.path.join(split_dir, split_file),
                "-out", blast_out,
                "-evalue", str(eval_param),
                "-outfmt", str(out_fmt),
                "-max_target_seqs", str(max_target_seqs),
            ]
            subprocess.run(cmd, check=True)
    return "blast_complete"


@python_app
def merge_blast_results_app(work_dir, merge_results_dir, db_dir, file_name):
    import os
    import socket
    print("Merge Blast Running on node:", socket.gethostname(), flush=True)
    db_list_path = os.path.join(db_dir, "db-list")
    with open(db_list_path) as f:
        databases = [line.strip() for line in f.readlines()]

    for db in databases:
        results_by_db = os.path.join(merge_results_dir, f"{db}.fasta")
        os.makedirs(results_by_db, exist_ok=True)
        blast_out_dir = os.path.join(work_dir, "results", "05C_blast", f"{db}.fasta", file_name)
        blast_results = os.path.join(results_by_db, f"{file_name}.txt")
        blast_gff = os.path.join(results_by_db, f"{file_name}.gff")

        with open(blast_results, "w") as outfile:
            for result_file in os.listdir(blast_out_dir):
                with open(os.path.join(blast_out_dir, result_file), "r") as infile:
                    outfile.write(infile.read())

        with open(blast_results, "r") as infile, open(blast_gff, "w") as outfile:
            for line in infile:
                fields = line.strip().split("\t")
                if len(fields) > 7:
                    gff_line = (
                        f"{fields[0]}\tblast\tgene\t{fields[6]}\t{fields[7]}"
                        f"\t.\t.\t.\tID=Gene{fields[6]};Name={fields[1]}\n"
                    )
                    outfile.write(gff_line)

    return os.path.join(
        merge_results_dir,
        "AVrC_allrepresentatives.fasta",
        "clusterRes_rep_seq.fasta.txt",
    )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class BLASTAgent(Agent):
    def __init__(self):
        parsl.clear()
        parsl.load(blast_config)

    @action
    async def run_full_blast(
        self, work_dir, split_size, results_dir, query_dir, cluster_file,
        db_dir, blast_results_dir, blast_type, eval_param, out_fmt,
        max_target_seqs, merge_results_dir, max_db_size, db_list_path,
    ):
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        fasta_file = next(
            (os.path.join(root, file)
             for root, _, files in os.walk(query_dir)
             for file in files if file.endswith(".fasta")),
            None,
        )
        if not fasta_file:
            raise FileNotFoundError("No .fasta file found in query directory.")
        file_name = os.path.basename(fasta_file)

        split_dir = os.path.join(query_dir, "fa_split")
        split_future = split_fasta_app(fasta_file, split_dir, split_size)
        split_result = await asyncio.to_thread(split_future.result)

        db_future = make_blast_db_app(db_dir, max_db_size, db_list_path)
        await asyncio.to_thread(db_future.result)

        blast_future = run_blast_app(
            split_result, blast_results_dir, db_dir,
            blast_type, eval_param, out_fmt, max_target_seqs,
        )
        await asyncio.to_thread(blast_future.result)

        merge_future = merge_blast_results_app(work_dir, merge_results_dir, db_dir, file_name)
        hits_file = await asyncio.to_thread(merge_future.result)

        # match_ratio: fraction of clustered contigs that hit the AVrC database.
        # Kept as a future-use evaluation signal alongside F1.
        cluster_contigs = set()
        with open(cluster_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    cluster_contigs.add(line[1:].strip())

        hits_contigs = set()
        with open(hits_file, "r") as f:
            for line in f:
                if line.strip():
                    hits_contigs.add(line.split()[0].strip())

        matching_contigs = cluster_contigs & hits_contigs
        match_ratio = len(matching_contigs) / len(cluster_contigs) if cluster_contigs else 0.0
        print(f"Total contigs in cluster: {len(cluster_contigs)}", flush=True)
        print(f"Contigs with BLAST hits:  {len(hits_contigs)}", flush=True)
        print(f"Matching:                 {len(matching_contigs)}", flush=True)
        print(f"match_ratio:              {match_ratio:.4f}", flush=True)
        return hits_file, match_ratio


# --------------------------------------------------------------------------- #
# Annotation (not a Parsl app — runs locally in the coordinator process)
# --------------------------------------------------------------------------- #

def annotate_blast(hits_file, annotations_dir, output_dir, script_path, pctid, length):
    import subprocess
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(annotations_dir):
        if file == ".DS_Store" or file.startswith("._"):
            os.remove(os.path.join(annotations_dir, file))

    for file in os.listdir(annotations_dir):
        ann_path = os.path.join(annotations_dir, file)
        if os.path.isfile(ann_path):
            out_file = os.path.join(output_dir, f"annotated_{file}")
            cmd = [
                script_path,
                "-b", hits_file,
                "-a", ann_path,
                "-o", out_file,
                "-p", pctid,
                "-l", length,
            ]
            print("Running:", " ".join(str(c) for c in cmd), flush=True)
            subprocess.run(cmd, check=True)

    return "Pipeline Complete."
