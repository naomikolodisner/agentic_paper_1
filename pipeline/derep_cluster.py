import asyncio
import parsl
from parsl import python_app
from academy.agent import Agent, action

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.parsl_configs import derep_cluster_config


# --------------------------------------------------------------------------- #
# Parsl apps
# --------------------------------------------------------------------------- #

@python_app
def dereplicate_app(
    sample_id, subset_spades, cluster_dir, cluster_res_derep,
    tmp_dir_derep, input_fasta, cleaned_fasta, out_derep,
):
    import os
    import subprocess
    import socket
    print("Dereplicate Running on node:", socket.gethostname(), flush=True)
    os.makedirs(cluster_dir, exist_ok=True)

    cmd_mmseqs_derep = [
        "conda", "run", "-n", "mmseqs2",
        "mmseqs", "easy-cluster", subset_spades, cluster_res_derep, tmp_dir_derep,
        "--min-seq-id", "0.99", "-c", "0.90", "--cov-mode", "1",
    ]
    subprocess.run(cmd_mmseqs_derep, check=True)

    cmd_awk = (
        r"""awk '/^>/{if($0!=prev){print; prev=$0}} !/^>/' """
        + input_fasta + f" > {cleaned_fasta}"
    )
    subprocess.run(cmd_awk, shell=True, check=True)

    os.makedirs(out_derep, exist_ok=True)
    done_flag = os.path.join(out_derep, f"done_{sample_id}.flag")
    with open(done_flag, "w") as f:
        f.write("done\n")

    return os.path.join(out_derep, "dereplicated.fasta")


@python_app
def cluster_app(
    sample_ids, out_derep, derep_fasta, out_cluster,
    cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst,
):
    import os
    import shutil
    import time
    import subprocess
    import socket
    print("Cluster Running on node:", socket.gethostname(), flush=True)

    done_flags = [os.path.join(out_derep, f"done_{sid}.flag") for sid in sample_ids]
    while not all(os.path.exists(flag) for flag in done_flags):
        time.sleep(5)

    with open(derep_fasta, "w") as outfile:
        for root, _, files in os.walk(out_derep):
            for file in files:
                if file.endswith("cleaned_clusterRes_all_seqs.fasta"):
                    with open(os.path.join(root, file)) as infile:
                        shutil.copyfileobj(infile, outfile)

    os.makedirs(out_cluster, exist_ok=True)
    cmd_mmseqs_cluster = [
        "conda", "run", "-n", "mmseqs2",
        "mmseqs", "easy-cluster", derep_fasta, cluster_res_cluster,
        tmp_dir_cluster, "--min-seq-id", "0.95", "-c", "0.75", "--cov-mode", "1",
    ]
    subprocess.run(cmd_mmseqs_cluster, check=True)

    os.makedirs(rep_seq_dst, exist_ok=True)
    shutil.copy(rep_seq_src, os.path.join(rep_seq_dst, "clusterRes_rep_seq.fasta"))
    return rep_seq_dst, rep_seq_src


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class DereplicationClusteringAgent(Agent):
    def __init__(self):
        parsl.clear()
        parsl.load(derep_cluster_config)

    @action
    async def run_dereplicate(
        self, sample_id, subset_spades, cluster_dir,
        cluster_res_derep, tmp_dir_derep, input_fasta,
        cleaned_fasta, out_derep,
    ):
        future = dereplicate_app(
            sample_id, subset_spades, cluster_dir,
            cluster_res_derep, tmp_dir_derep,
            input_fasta, cleaned_fasta, out_derep,
        )
        return await asyncio.to_thread(future.result)

    @action
    async def run_cluster(
        self, sample_ids, out_derep, derep_fasta,
        out_cluster, cluster_res_cluster, tmp_dir_cluster,
        rep_seq_src, rep_seq_dst,
    ):
        future = cluster_app(
            sample_ids, out_derep, derep_fasta, out_cluster,
            cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst,
        )
        return await asyncio.to_thread(future.result)
