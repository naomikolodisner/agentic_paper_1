import os
import subprocess
import shutil
from academy.manager import Manager
import asyncio
from concurrent.futures import ThreadPoolExecutor
from academy.manager import Manager
from academy.exchange.local import LocalExchangeFactory
import time
import parsl
import asyncio
from parsl import python_app
from academy.agent import Agent, action, loop
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.usage_tracking.levels import LEVEL_1
import random
import asyncio
from datetime import datetime

viral_config = Config(
     executors=[
          HighThroughputExecutor(
               label="Parsl_htex",
               worker_debug=False,
               cores_per_worker=1.0,
               max_workers_per_node=94,
               provider=SlurmProvider(
                    partition='windfall',
                    #account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=7,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='40:00:00',
                    launcher=SrunLauncher(),
                    worker_init='',
               ),
          )
     ],
     usage_tracking=LEVEL_1,
)

checkv_config = Config(
     executors=[
          HighThroughputExecutor(
               label="Parsl_htex",
               worker_debug=False,
               cores_per_worker=1.0,
               max_workers_per_node=94,
               provider=SlurmProvider(
                    partition='windfall',
                    #account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=7,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='40:00:00',
                    launcher=SrunLauncher(),
                    worker_init='',
               ),
          )
     ],
     usage_tracking=LEVEL_1,
)

derep_cluster_config = Config(
     executors=[
          HighThroughputExecutor(
               label="Parsl_htex",
               worker_debug=False,
               cores_per_worker=1.0,
               max_workers_per_node=94,
               provider=SlurmProvider(
                    partition='windfall',
                    #account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=7,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='40:00:00',
                    launcher=SrunLauncher(),
                    worker_init='',
               ),
          )
     ],
     usage_tracking=LEVEL_1,
)

blast_config = Config(
     executors=[
          HighThroughputExecutor(
               label="Parsl_htex",
               worker_debug=False,
               cores_per_worker=1.0,
               max_workers_per_node=94,
               provider=SlurmProvider(
                    partition='windfall',
                    #account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=7,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='40:00:00',
                    launcher=SrunLauncher(),
                    worker_init='',
               ),
          )
     ],
     usage_tracking=LEVEL_1,
)

# === Turn config file into a dictionary of variables ===
'''
def make_config(config_file):
    config = {}
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):]  # Remove 'export '
            if '=' in line:
                key, val = line.split('=', 1)
                config[key.strip()] = val.strip().strip('"').strip("'")
                val = os.path.expandvars(os.path.expanduser(val))
                config[key.strip()] = val
    return config

def read_sample_ids(sample_ids_file):
    # Read sample IDs from the file
    with open(sample_ids_file, "r") as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    return sample_ids


@python_app
def unzip_fasta_app(spades_gz, unzipped_spades_path):
    import subprocess
    import os
    import socket

    print("Unzip Running on node:", socket.gethostname(), flush=True)
    if os.path.exists(unzipped_spades_path):
        print(f"[INFO] File already exists: {unzipped_spades_path}", flush=True)
        return unzipped_spades_path

    try:
        subprocess.run(["gzip", "-dk", spades_gz], check=True)
        print(f"[INFO] Successfully unzipped: {spades_gz}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to unzip {spades_gz}: {e}", flush=True)
        raise e

    return unzipped_spades_path

# === Subprocess Wrapper ===

async def run_subprocess(cmd, **kwargs):
    try:
        return await asyncio.to_thread(subprocess.run, cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print("Command failed:", e.cmd, flush=True)
        print("Return code:", e.returncode, flush=True)
        print("Output:", e.output, flush=True)
        print("Stderr:", e.stderr, flush=True)
       

# === Viral Detection Agent ===

@python_app
def virsorter2_app(unzipped_spades, virsorter2_output_dir):
    import subprocess
    import os
    import socket
    import shutil
    print("VirSorter2 Running on node:", socket.gethostname(), flush=True)
    if os.path.exists(virsorter2_output_dir):
        shutil.rmtree(virsorter2_output_dir)
    os.makedirs(virsorter2_output_dir)   
    cmd = [
        "conda", "run", "-n", "virsorter2_env",
        "virsorter", "run", "-w", virsorter2_output_dir,
        "-i", unzipped_spades, "--min-length", "1500", "-j", "4", "all"
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(virsorter2_output_dir, "final-viral-combined.fa")

@python_app
def deepvirfinder_app(unzipped_spades, dvf_output_dir, dvf_db, work_dir, script_dir):
    import subprocess
    import os
    os.chdir(dvf_db)
    import socket
    print("DeepVirFinder Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "dvf_env",
        "python", "dvf.py",
        "-i", unzipped_spades,
        "-o", dvf_output_dir,
        "-l", "1500"
    ]
    subprocess.run(cmd, check=True)

    os.chdir(work_dir)

    dvf_output = os.path.join(dvf_output_dir, "contigs.fasta_gt1500bp_dvfpred.txt")
    dvf_fasta_output = os.path.join(dvf_output_dir, "dvf.fasta")

    os.chdir(script_dir)

    cmd2 = [
        "conda", "run", "-n", "dvf_env",
        "python", "id_from_fasta.py",
        "-c", unzipped_spades,
        "-d", dvf_output,
        "-o", dvf_fasta_output
    ]
    subprocess.run(cmd2, check=True)

    os.chdir(work_dir)
    return dvf_fasta_output

@python_app
def genomad_app(unzipped_spades, genomad_output_dir, genomad_db):
    import subprocess
    import os
    import socket
    print("GeNomad Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "genomad_env",
        "genomad", "end-to-end", "--cleanup", "--restart", 
        unzipped_spades, genomad_output_dir, genomad_db
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(genomad_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def marvel_app(unzipped_spades, marvel_output_dir, marvel_db):
    import subprocess
    import os
    import socket
    print("MARVEL Running on node:", socket.gethostname(), flush=True)
    os.chdir(marvel_db)
    cmd = [
        "conda", "run", "-n", "marvel_env", "python3",
        "marvel_bins.py", "-i", unzipped_spades, "-t", "16",
        "-o", marvel_output_dir]
    subprocess.run(cmd, check=True)
    #find out what output directory iates   
    return os.path.join(marvel_output_dir)

@python_app
def virfinder_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("VirFinder Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "virfinder_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(virfinder_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def vibrant_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("VIBRANT Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "vibrant_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(vibrant_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def viralverify_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("viralVerify Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "viralverify_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(viralverify_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def viraminer_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("ViraMiner Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "viraminer_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(viraminer_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def metaphinder_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("MetaPhinder Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "metaphinder_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(metaphinder_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def seeker_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("Seeker Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "seeker_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(seeker_output_dir, "contigs_summary", "contigs_virus.fna")

@python_app
def virsorter_app(input1, input2, input3):
    import subprocess
    import os
    import socket
    print("VirSorter Running on node:", socket.gethostname(), flush=True)
    cmd = [
        "conda", "run", "-n", "virsorter_env"]
    subprocess.run(cmd, check=True)
    return os.path.join(virsorter_output_dir, "contigs_summary", "contigs_virus.fna")

class ViralDetectionAgent(Agent):
    def __init__(self):
        # Load config once on creation
        parsl.clear()
        parsl.load(viral_config)

    @action
    async def unzip_fasta(self, spades_gz: str, unzipped_spades_path: str) -> str:
        future = unzip_fasta_app(spades_gz, unzipped_spades_path)
        result = await asyncio.to_thread(future.result)
        return result

    @action
    async def run_genomad(self, unzipped_spades: str, genomad_output_dir: str, genomad_db: str) -> str:
        future = genomad_app(unzipped_spades, genomad_output_dir, genomad_db)
        return await asyncio.to_thread(future.result)

    @action
    async def run_virsorter(self, unzipped_spades: str, virsorter_output_dir: str) -> str:
        future = virsorter_app(unzipped_spades, virsorter_output_dir)
        return await asyncio.to_thread(future.result)

    @action
    async def run_deepvirfinder(self, unzipped_spades: str, dvf_output_dir: str, 
                                dvf_db: str, work_dir: str, script_dir: str) -> str:
        future = deepvirfinder_app(unzipped_spades, dvf_output_dir, dvf_db, work_dir, script_dir)
        return await asyncio.to_thread(future.result)

    @action
    async def run_marvel(self, unzipped_spades: str, marvel_output_dir: str, marvel_db: str) -> str:
        future = marvel_app(unzipped_spades, marvel_output_dir, marvel_db)
        return await asyncio.to_thread(future.result)

    @action
    async def run_tool(self, tool, unzipped_spades: str, genomad_output_dir: str, genomad_db: str, 
                       virsorter_output_dir: str, dvf_output_dir: str, dvf_db: str,
                       work_dir: str, script_dir: str, marvel_output_dir: str, marvel_db: str) -> str:
        
        if tool == "GeNomad":
            result = await self.run_genomad(unzipped_spades, genomad_output_dir, genomad_db)
        elif tool == "VirSorter2":
            result =  await self.run_virsorter(unzipped_spades, virsorter_output_dir)
        elif tool == "MARVEL":
            result == await self.run_marvel(unzipped_spades, marvel_output_dir, marvel_db)
        else:
            result = await self.run_deepvirfinder(unzipped_spades, dvf_output_dir, dvf_db, work_dir, script_dir)
        return result

# Parsl Python app for CheckV
@python_app
def checkv_app(checkv_parser, parse_length, work_dir,
               unzipped_spades, viral_result, checkv_output_dir,
               parse_input, selection_csv, checkvdb):
    import os
    import subprocess
    import socket
    import shutil
    print("CheckV Running on node:", socket.gethostname(), flush=True)
    if os.path.exists(checkv_output_dir):
        shutil.rmtree(checkv_output_dir)
    os.makedirs(checkv_output_dir)

    cmd_checkv = [
        "conda", "run", "-n", "checkv_env", "checkv", "end_to_end",
        viral_result, checkv_output_dir, "-t", "4", "-d", checkvdb
    ]
    cmd_parser = [
        "conda", "run", "-n", "r_env", "Rscript", checkv_parser,
        "-i", parse_input, "-l", parse_length, "-o", selection_csv
    ]

    subprocess.run(cmd_checkv, check=True)
    subprocess.run(cmd_parser, check=True)
    cleaned_selection_csv = selection_csv.replace(".csv", "_cleaned.csv")
    with open(selection_csv, "r") as infile, open(cleaned_selection_csv, "w") as outfile:
        for line in infile:
            if line.startswith("contig_id"):
                outfile.write(line)
            else:
                clean_line = line.split("||")[0].strip()
                outfile.write(f"{clean_line}\n")
    cmd_seqtk = [
        "conda", "run", "-n", "seqtk_env", "seqtk", "subseq",
        unzipped_spades, cleaned_selection_csv
    ]
    subset_spades = os.path.join(checkv_output_dir, "subset_spades.fasta")
    with open(subset_spades, "w") as out_f:
        subprocess.run(cmd_seqtk, check=True, stdout=out_f)

    os.chdir(work_dir)
    total = 0
    high_quality = 0
    quality = os.path.join(checkv_output_dir, "quality_summary.tsv")
    with open(quality, 'r') as f:
        for line in f:
            # Skip the header line
            if line.startswith("contig_id") or line.strip() == "":
                continue
            columns = line.strip().split('\t')

            if len(columns) < 8:
                continue  # Skip incomplete lines

            quality = columns[7].strip()
            total += 1
            if quality == "High-quality":
                high_quality += 1
    if total == 0:
        quality_ratio =  0.0
    quality_ratio =  high_quality / total

    return subset_spades, quality_ratio

# Agent class using the app
class CheckVAgent(Agent):
    def __init__(self):
        # Load config once on creation
        parsl.clear()
        parsl.load(checkv_config)  
        
    @action
    async def run_checkv(
        self, checkv_parser: str, parse_length: str, work_dir: str,
        unzipped_spades: str, viral_result: str, checkv_output_dir: str,
        parse_input: str, selection_csv: str, checkvdb: str
    ) -> str:

        # Run the Parsl app
        future = checkv_app(
            checkv_parser, parse_length, work_dir,
            unzipped_spades, viral_result, checkv_output_dir,
            parse_input, selection_csv, checkvdb
        )

        # Await the result asynchronously
        subset_spades, quality_ratio = await asyncio.to_thread(future.result)

        return subset_spades, quality_ratio

# === Dereplication/Clustering Agent ===

# === Parsl Apps ===

@python_app
def dereplicate_app(
    sample_id, subset_spades, cluster_dir, cluster_res_derep,
    tmp_dir_derep, input_fasta, cleaned_fasta, out_derep
):
    import os
    import subprocess
    import socket
    print("Dereplicate Running on node:", socket.gethostname(), flush=True)
    os.makedirs(cluster_dir, exist_ok=True)

    cmd_mmseqs_derep = [
        "conda", "run", "-n", "mmseqs2_env",
        "mmseqs", "easy-cluster", subset_spades, cluster_res_derep, tmp_dir_derep,
        "--min-seq-id", "0.99", "-c", "0.90", "--cov-mode", "1"
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
    cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst
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

    with open(derep_fasta, 'w') as outfile:
        for root, _, files in os.walk(out_derep):
            for file in files:
                if file.endswith("cleaned_clusterRes_all_seqs.fasta"):
                    with open(os.path.join(root, file)) as infile:
                        shutil.copyfileobj(infile, outfile)

    os.makedirs(out_cluster, exist_ok=True)
    cmd_mmseqs_cluster = [
        "conda", "run", "-n", "mmseqs2_env",
        "mmseqs", "easy-cluster", derep_fasta, cluster_res_cluster,
        tmp_dir_cluster, "--min-seq-id", "0.95", "-c", "0.75", "--cov-mode", "1"
    ]
    subprocess.run(cmd_mmseqs_cluster, check=True)

    os.makedirs(rep_seq_dst, exist_ok=True)
    shutil.copy(rep_seq_src, os.path.join(rep_seq_dst, "clusterRes_rep_seq.fasta"))
    return rep_seq_dst, rep_seq_src

# === Agent ===

class DereplicationClusteringAgent(Agent):
    def __init__(self):
        # Load config once on creation
        parsl.clear()
        parsl.load(derep_cluster_config)

    @action
    async def run_dereplicate(
        self, sample_id: str, subset_spades: str, cluster_dir: str,
        cluster_res_derep: str, tmp_dir_derep: str, input_fasta: str,
        cleaned_fasta: str, out_derep: str
    ) -> str:

        future = dereplicate_app(
            sample_id, subset_spades, cluster_dir,
            cluster_res_derep, tmp_dir_derep,
            input_fasta, cleaned_fasta, out_derep
        )
        result = await asyncio.to_thread(future.result)
        return result

    @action
    async def run_cluster(
        self, sample_ids: list[str], out_derep: str, derep_fasta: str,
        out_cluster: str, cluster_res_cluster: str, tmp_dir_cluster: str,
        rep_seq_src: str, rep_seq_dst: str
    ) -> str:

        future = cluster_app(
            sample_ids, out_derep, derep_fasta, out_cluster,
            cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst
        )
        rep_seq_dst, rep_seq_src = await asyncio.to_thread(future.result)
        return rep_seq_dst, rep_seq_src


# === Python App: Split FASTA ===
@python_app
def split_fasta_app(fasta_file: str, split_dir: str, split_size: int):
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
        "faSplit", "about", fasta_file, str(split_size), f"{split_dir}/"
    ]
    subprocess.run(cmd, check=True)
    return split_dir


# === Python App: Make BLAST DB ===
@python_app
def make_blast_db_app(db_dir: str, max_db_size: int, db_list_path: str):
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
                "-max_file_sz", str(max_db_size)
            ]
            subprocess.run(cmd, check=True)

    return "blast_db_complete"


# === Python App: Run BLAST ===
@python_app
def run_blast_app(
    split_dir, blast_results_dir, db_dir,
    blast_type, eval_param, out_fmt, max_target_seqs
):
    import os
    import subprocess
    import socket
    print("BLAST Running on node:", socket.gethostname(), flush=True)
    split_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".fa"))
    db_list_path = os.path.join(db_dir, "db-list")

    with open(db_list_path, 'r') as f:
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
                "-max_target_seqs", str(max_target_seqs)
            ]
            subprocess.run(cmd, check=True)

    return "blast_complete"


# === Python App: Merge Results ===
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

        with open(blast_results, 'w') as outfile:
            for result_file in os.listdir(blast_out_dir):
                with open(os.path.join(blast_out_dir, result_file), 'r') as infile:
                    outfile.write(infile.read())

        with open(blast_results, 'r') as infile, open(blast_gff, 'w') as outfile:
            for line in infile:
                fields = line.strip().split('\t')
                if len(fields) > 7:
                    gff_line = f"{fields[0]}\tblast\tgene\t{fields[6]}\t{fields[7]}\t.\t.\t.\tID=Gene{fields[6]};Name={fields[1]}\n"
                    outfile.write(gff_line)

    return os.path.join(merge_results_dir, "AVrC_allrepresentatives.fasta", "clusterRes_rep_seq.fasta.txt")


# === Agent: Blast ===
class BLASTAgent(Agent):
    def __init__(self):
        # Load config once on creation
        parsl.clear()
        parsl.load(blast_config)

    @action
    async def run_full_blast(
        self, work_dir: str, split_size: int,
        results_dir: str, query_dir: str, cluster_file: str, db_dir: str,
        blast_results_dir: str, blast_type: str, eval_param: float,
        out_fmt: int, max_target_seqs: int, merge_results_dir: str,
        max_db_size: int, db_list_path: str
    ) -> str:
        # Clean output dirs
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Find FASTA file
        fasta_file = next(
            (os.path.join(root, file)
             for root, _, files in os.walk(query_dir)
             for file in files if file.endswith(".fasta")),
            None
        )
        if not fasta_file:
            raise FileNotFoundError("No .fasta file found in query directory.")
        file_name = os.path.basename(fasta_file)

        # 1. Split FASTA
        split_dir = os.path.join(query_dir, "fa_split")
        split_future = split_fasta_app(fasta_file, split_dir, split_size)
        split_result = await asyncio.to_thread(split_future.result)

        # 2. Make BLAST DB
        db_future = make_blast_db_app(db_dir, max_db_size, db_list_path)
        await asyncio.to_thread(db_future.result)

        # 3. Run BLAST
        blast_future = run_blast_app(split_result, blast_results_dir, db_dir, blast_type, eval_param, out_fmt, max_target_seqs)
        await asyncio.to_thread(blast_future.result)

        # 4. Merge
        merge_future = merge_blast_results_app(work_dir, merge_results_dir, db_dir, file_name)
        hits_file = await asyncio.to_thread(merge_future.result)
       
        # Get all headers from the FASTA file
        cluster_contigs = set()
        with open(cluster_file, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    header = line[1:].strip()
                    cluster_contigs.add(header)

        # Get all contig names from the TXT file
        hits_contigs = set()
        with open(hits_file, 'r') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    contig = line.split()[0].strip()
                    hits_contigs.add(contig)

        # Calculate intersection
        matching_contigs = cluster_contigs & hits_contigs

        # Print stats
        print(f"Total contigs in FASTA: {len(cluster_contigs)}", flush=True)
        print(f"Contigs in TXT file: {len(hits_contigs)}", flush=True)
        print(f"Matching contigs: {len(matching_contigs)}", flush=True)
        match_ratio = len(matching_contigs) / len(cluster_contigs) 
        print(f"Fraction matched: {match_ratio:.4f}", flush=True)
        return hits_file, match_ratio


# === Annotation ===

def annotate_blast(hits_file, annotations_dir, output_dir, script_path, pctid, length):

    """
    Loops over all annotation CSV files and runs the annotation script using subprocess.run.
    Replicates shell logic: looping over files, constructing arguments, and running script.
    """
    import subprocess
    import os
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Remove unwanted files
    for file in os.listdir(annotations_dir):
        if file == '.DS_Store' or file.startswith('._'):
            os.remove(os.path.join(annotations_dir, file))

    # Debug print of annotation files
    print("Files in annotation directory:", flush=True)
    for f in os.listdir(annotations_dir):
        print(os.path.join(annotations_dir, f), flush=True)

    # Loop over each annotation file
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
                "-l", length
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    final = "Pipeline Complete."
    return final

class ToolSelector:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.best_tool = None
        self.best_score = -1.0
        self.tool_scores = {}

    def evaluate_tool(self, avg_quality_ratio: float, match_ratio: float) -> float:
        """Combine CheckV and BLAST metrics into a single score."""
        return (avg_quality_ratio + match_ratio) / 2

    def update_tool_score(self, tool_name: str, avg_quality_ratio: float, match_ratio: float):
        """Update score for a tool and track best tool so far."""
        score = self.evaluate_tool(avg_quality_ratio, match_ratio)
        self.tool_scores[tool_name] = score
        if score > self.best_score:
            self.best_score = score
            self.best_tool = tool_name

    def choose_tool(self, available_tools: list[str]) -> str:
        """Choose the next tool using alpha-greedy strategy."""
        if not self.best_tool or self.best_tool not in available_tools:
            return random.choice(available_tools)

        if random.random() < self.alpha:
            return self.best_tool
        else:
            return random.choice(available_tools)

class CoordinatorAgent(Agent):
    def __init__(self, viral_handle, checkv_handle, cluster_handle, blast_handle, config_path, shutdown_event):
        super().__init__()
        self.viral_handle = viral_handle
        self.checkv_handle = checkv_handle
        self.cluster_handle = cluster_handle
        self.blast_handle = blast_handle
        self.config = make_config(config_path)
        self.selector = ToolSelector(alpha=0.6)
        #self.current_tool = random.choice(["GeNomad", "VirSorter2", "DeepVirFinder", "MARVEL"])
        self.current_tool = "MARVEL"
        self.quality_ratios_history = []
        self.match_ratios_history = []
        self.shutdown = shutdown_event
    @loop
    async def continuous_pipeline(self, shutdown: asyncio.Event) -> None:
        round_count = 0
        while not shutdown.is_set():
            print(f"\n=== [Coordinator] Starting round {round_count+1} with tool: {self.current_tool} ===", flush=True)
            sample_ids_file = os.path.join(self.config['XFILE_DIR'], self.config['XFILE'])
            sample_ids = read_sample_ids(sample_ids_file)
            first_sample_id = sample_ids[0]
            # Run per-sample
            per_sample_tasks = [
                asyncio.create_task(
                    process_sample(sid, self.config, self.current_tool,
                                   self.viral_handle, self.checkv_handle, self.cluster_handle, first_sample_id)
                )
                for sid in sample_ids
            ]
            results = await asyncio.gather(*per_sample_tasks)
            # Process quality ratios
            quality_ratios = [qr for _, qr in results if qr is not None]
            avg_quality_ratio = sum(quality_ratios) / len(quality_ratios) if quality_ratios else 0.0
            self.quality_ratios_history.append(avg_quality_ratio)
            print(f"[Coordinator] Average quality ratio: {avg_quality_ratio:.4f}", flush=True)
            # Find derep FASTA
            derep_fasta = next((df for df, _ in results if df is not None), None)
            if derep_fasta is None:
                print("[Coordinator] Skipping round: no dereplicated fasta.", flush=True)
                await asyncio.sleep(10)
                continue
            # === Cluster ===
            out_cluster = self.config["OUT_CLUSTER"]
            work_dir = self.config["WORK_DIR"]
            cluster_res_cluster = os.path.join(out_cluster, "clusterRes")
            tmp_dir_cluster = os.path.join(out_cluster, "tmp")
            rep_seq_src = os.path.join(out_cluster, "clusterRes_rep_seq.fasta")
            rep_seq_dst = os.path.join(work_dir, "query")
            out_derep = self.config["OUT_DEREP"]
            query_dir, cluster_file = await(await self.cluster_handle.run_cluster(
                sample_ids, out_derep, derep_fasta, out_cluster,
                cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst
            ))
            # === BLAST ===
            db_dir = self.config["DB_DIR"]
            max_db_size = self.config["MAX_DB_SIZE"]
            db_list_path = os.path.join(db_dir, "db-list")
            prog = "05B_launchblast"
            fasta_dir = self.config["FASTA_DIR"]
            split_size = self.config["FA_SPLIT_FILE_SIZE"]
            results_dir = os.path.join(work_dir, "results_testing", prog)
            files_list_path = os.path.join(fasta_dir, "fasta-files")
            blast_results_dir = os.path.join(work_dir, "results_testing", "05C_blast")
            blast_type = self.config["BLAST_TYPE"]
            eval_param = self.config["EVAL"]
            out_fmt = self.config["OUT_FMT"]
            max_target_seqs = self.config["MAX_TARGET_SEQS"]
            merge_results_dir = os.path.join(work_dir, "results_testing", "05D_mergeblast")
            hits_file, match_ratio = await(await self.blast_handle.run_full_blast(
                work_dir, split_size, results_dir, query_dir, cluster_file, db_dir,
                blast_results_dir, blast_type, eval_param, out_fmt,
                max_target_seqs, merge_results_dir, max_db_size, db_list_path
            ))
            self.match_ratios_history.append(match_ratio)
            print(f"[Coordinator] BLAST Match Ratio: {match_ratio:.4f}", flush=True)
            # === Update Tool Score and Select Next Tool ===
            self.selector.update_tool_score(self.current_tool, avg_quality_ratio, match_ratio)
            self.current_tool = self.selector.choose_tool(["VirSorter2", "DeepVirFinder", "GeNomad", "MARVEL"])
            print(f"[Coordinator] Selected tool for next round: {self.current_tool}", flush=True)
            # === Annotation ===
            annotations_dir = self.config['ANNOTATIONS']
            out_annotate = self.config['OUTPUT']
            script_path = os.path.join(self.config['PROJECT_ROOT'], "solution1_manual.py")
            pctid = self.config['PCTID']
            length = self.config['LENGTH']
            final = annotate_blast(hits_file, annotations_dir, out_annotate, script_path, pctid, length)
            print("[Coordinator] Final annotation summary:", final, flush=True)
            round_count += 1
            if round_count >= 10:  # Stop after 10 rounds
                print("[Coordinator] Completed 10 rounds. Initiating shutdown.", flush=True)
                self.shutdown.set()
                final_avg_quality = sum(self.quality_ratios_history) / len(self.quality_ratios_history) if self.quality_ratios_history else 0.0
                final_avg_match = sum(self.match_ratios_history) / len(self.match_ratios_history) if self.match_ratios_history else 0.0
                print(f"\n[Coordinator] FINAL average quality ratio over 10 rounds: {final_avg_quality:.4f}", flush=True)
                print(f"[Coordinator] FINAL average match ratio over 10 rounds: {final_avg_match:.4f}\n", flush=True)   
                print(f"Best Tool: {self.selector.best_tool}", flush=True)
                end_time = datetime.datetime.now()
                print(f"[Main] End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                break
            await asyncio.sleep(5)  # wait between rounds

# Helper for per-sample pipeline
async def process_sample(sample_id, config, tool, viral_handle, checkv_handle, cluster_handle, first_sample_id):   
    # === Unzip ===
    spades_gz = os.path.join(config['SPADES_DIR'], sample_id, "contigs.fasta.gz")
    unzipped_spades_path = os.path.join(config['SPADES_DIR'], sample_id, "contigs.fasta")
    unzipped_spades = (await viral_handle.unzip_fasta(spades_gz, unzipped_spades_path))
    # === Viral Detection ===
    genomad_output_dir = os.path.join(config['OUT_GENOMAD'], sample_id)
    genomad_db = config["GENOMAD_DB"]
    virsorter_output_dir = os.path.join(config["OUT_VIRSORT"], sample_id)
    dvf_output_dir = os.path.join(config["OUT_DVF"], sample_id)
    dvf_db = config["DVF_DB"]
    work_dir = config["WORK_DIR"]
    script_dir = work_dir
    marvel_output_dir = os.path.join(config["OUT_MARVEL"], sample_id)
    marvel_db = os.path.join(config["MARVEL_DB"])
    start_time = datetime.now()  
    viral_result = await(await viral_handle.run_tool(tool, unzipped_spades, genomad_output_dir, genomad_db, virsorter_output_dir,
                                                      dvf_output_dir, dvf_db, work_dir, script_dir, marvel_output_dir, marvel_db))
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"{tool} started at {start_time} and ended at {end_time} - duration: {elapsed}", flush=True)    
 
    # === CheckV ===
    checkv_parser = config["CHECKV_PARSER"]
    parse_length = str(config["PARSE_LENGTH"])
    work_dir = config["WORK_DIR"]
    checkv_output_dir = os.path.join(config["OUT_CHECKV"], sample_id)
    parse_input = os.path.join(checkv_output_dir, "contamination.tsv")
    selection_csv = os.path.join(checkv_output_dir, "selection2_viral.csv")
    checkvdb = config["CHECKVDB"]
    subset_spades, quality_ratio = await(await checkv_handle.run_checkv(
        checkv_parser, parse_length, work_dir, unzipped_spades, viral_result,
        checkv_output_dir, parse_input, selection_csv, checkvdb))
    print(quality_ratio, flush=True)
    # === Dereplication ===
    cluster_dir = os.path.join(config["OUT_DEREP"], sample_id)
    cluster_res_derep = os.path.join(cluster_dir, "clusterRes")
    tmp_dir_derep = os.path.join(cluster_dir, "tmp")
    input_fasta = f"{cluster_res_derep}_all_seqs.fasta"
    cleaned_fasta = os.path.join(cluster_dir, "cleaned_clusterRes_all_seqs.fasta")
    out_derep = config["OUT_DEREP"]
    derep_fasta = await(await cluster_handle.run_dereplicate(
        sample_id, subset_spades, cluster_dir, cluster_res_derep,
        tmp_dir_derep, input_fasta, cleaned_fasta, out_derep))
    return derep_fasta if sample_id == first_sample_id else None, quality_ratio

async def main():
    start_time = datetime.now()
    print(f"[Main] Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    shutdown_event = asyncio.Event()
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor()
    ) as manager:
        viral_handle = await manager.launch(ViralDetectionAgent())
        checkv_handle = await manager.launch(CheckVAgent())
        cluster_handle = await manager.launch(DereplicationClusteringAgent())
        blast_handle = await manager.launch(BLASTAgent())
        config_path = os.path.join(os.getcwd(), "config_py.sh")
        coordinator = await manager.launch(
            CoordinatorAgent,
            args=(viral_handle, checkv_handle, cluster_handle, blast_handle, config_path, shutdown_event)
        )
        await shutdown_event.wait()
        print("[Main] Shutdown complete.")
    end_time = datetime.now()
    print(f"[Main] End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[Main] Total runtime: {str(end_time - start_time)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())


