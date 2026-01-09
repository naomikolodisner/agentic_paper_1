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
from academy.agent import Agent, action
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.usage_tracking.levels import LEVEL_1
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from academy.manager import Manager
from academy.exchange.local import LocalExchangeFactory


viral_config = Config(
     executors=[
          HighThroughputExecutor(
               label="Parsl_htex",
               worker_debug=False,
               cores_per_worker=1.0,
               max_workers_per_node=94,
               provider=SlurmProvider(
                    partition='standard',
                    account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=1,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='24:00:00',
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
                    partition='standard',
                    account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=1,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='24:00:00',
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
                    partition='standard',
                    account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=1,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='24:00:00',
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
                    partition='standard',
                    account='bhurwitz',
                    init_blocks=1,
                    mem_per_node=80,
                    cores_per_node=94,
                    nodes_per_block=1,
                    scheduler_options='',
                    cmd_timeout=60,
                    walltime='24:00:00',
                    launcher=SrunLauncher(),
                    worker_init='',
               ),
          )
     ],
     usage_tracking=LEVEL_1,
)

# === Turn config file into a dictionary of variables ===

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

    if os.path.exists(unzipped_spades_path):
        print(f"[INFO] File already exists: {unzipped_spades_path}")
        return unzipped_spades_path

    try:
        subprocess.run(["gzip", "-dk", spades_gz], check=True)
        print(f"[INFO] Successfully unzipped: {spades_gz}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to unzip {spades_gz}: {e}")
        raise e

    return unzipped_spades_path

# === Subprocess Wrapper ===

async def run_subprocess(cmd, **kwargs):
    try:
        return await asyncio.to_thread(subprocess.run, cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print("Command failed:", e.cmd)
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Stderr:", e.stderr)
       

# === Viral Detection Agent ===

@python_app
def virsorter_app(unzipped_spades, virsorter_output_dir):
    import subprocess
    import os
    cmd = [
        "conda", "run", "-n", "virsorter2_env",
        "virsorter", "run", "-w", virsorter_output_dir,
        "-i", unzipped_spades, "--min-length", "1500", "-j", "4", "all"
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(virsorter_output_dir, "final-viral-combined.fa")

@python_app
def deepvirfinder_app(unzipped_spades, dvf_output_dir, dvf_db, work_dir, script_dir):
    import subprocess
    import os
    os.chdir(dvf_db)

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
def genomad_app(unzipped_spades, genomad_output_dir, db):
    import subprocess
    import os
    cmd = [
        "conda", "run", "-n", "genomad_env",
        "genomad", "end-to-end", "--cleanup", 
        unzipped_spades, genomad_output_dir, db
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(output_dir, "contigs_summary", "contigs_virus.fna")

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
    async def run_genomad(self, unzipped_spades: str, genomad_output_dir: str, db: str) -> str:
        future = genomad_app(unzipped_spades, genomad_output_dir, db)
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
    async def run_tool(self, tool, unzipped_spades: str, genomad_output_dir: str, db: str, 
                       virsorter_output_dir: str, dvf_output_dir: str, dvf_db: str, 
                       work_dir: str, script_dir: str) -> str:
        
        if tool == "GeNomad":
            result = await self.run_genomad(unzipped_spades, genomad_output_dir, db)
        elif tool == "VirSorter2":
            result =  await self.run_virsorter(unzipped_spades, virsorter_output_dir)
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

    os.makedirs(checkv_output_dir, exist_ok=True)

    cmd_checkv = [
        "conda", "run", "-n", "checkv_env", "checkv", "end_to_end",
        viral_result, checkv_output_dir, "-t", "4", "-d", checkvdb
    ]
    cmd_parser = [
        "conda", "run", "-n", "r_env", "Rscript", checkv_parser,
        "-i", parse_input, "-l", parse_length, "-o", selection_csv
    ]
    cmd_seqtk = [
        "conda", "run", "-n", "seqtk_env", "seqtk", "subseq",
        unzipped_spades, selection_csv
    ]

    subprocess.run(cmd_checkv, check=True)
    subprocess.run(cmd_parser, check=True)

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

    os.makedirs(db_dir, exist_ok=True)
    os.chdir(db_dir)

    with open(db_list_path, "w") as db_list:
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".fasta"):
                    rel_path = os.path.join(root, file).lstrip("./")
                    db_list.write(rel_path + "\n")

    if not os.path.exists(db_list_path) or os.path.getsize(db_list_path) == 0:
        raise FileNotFoundError(f"Empty or missing db list: {db_list_path}")

    with open(db_list_path) as f:
        for line in f:
            db_file = line.strip()
            db_name = os.path.splitext(os.path.basename(db_file))[0]
            db_prefix = os.path.join(db_dir, db_name)

            if all(os.path.exists(f"{db_prefix}.{ext}") for ext in ["nhr", "nin", "nsq"]):
                continue

            cmd = [
                "conda", "run", "-n", "blast_env",
                "makeblastdb",
                "-title", db_name,
                "-out", db_prefix,
                "-in", db_file,
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

    split_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".fa"))
    db_list_path = os.path.join(db_dir, "db-list")

    with open(db_list_path, 'r') as f:
        databases = [line.strip() for line in f.readlines()]

    for split_file in split_files:
        for db in databases:
            result_dir = os.path.join(blast_results_dir, db, split_file)
            os.makedirs(result_dir, exist_ok=True)
            blast_out = os.path.join(result_dir, f"{split_file}.blastout")
            blast_db = os.path.join(db_dir, db)

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

    db_list_path = os.path.join(db_dir, "db-list")
    with open(db_list_path) as f:
        databases = [line.strip() for line in f.readlines()]

    for db in databases:
        results_by_db = os.path.join(merge_results_dir, db)
        os.makedirs(results_by_db, exist_ok=True)
        blast_out_dir = os.path.join(work_dir, "results", "05C_blast", db, file_name)
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
        print(f"Total contigs in FASTA: {len(cluster_contigs)}")
        print(f"Contigs in TXT file: {len(hits_contigs)}")
        print(f"Matching contigs: {len(matching_contigs)}")
        match_ratio = len(matching_contigs) / len(cluster_contigs) 
        print(f"Fraction matched: {match_ratio:.4f}")
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
    print("Files in annotation directory:")
    for f in os.listdir(annotations_dir):
        print(os.path.join(annotations_dir, f))

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
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    final = "Pipeline Complete."
    return final

# Helper for per-sample pipeline
async def process_sample(sample_id, config, tool, viral_handle, checkv_handle, cluster_handle, first_sample_id):
    # === Unzip ===
    spades_gz = os.path.join(config['SPADES_DIR'], sample_id, "contigs.fasta.gz")
    unzipped_spades_path = os.path.join(config['SPADES_DIR'], sample_id, "contigs.fasta")
    unzipped_spades = await(await viral_handle.unzip_fasta(spades_gz, unzipped_spades_path))

    # === GeNomad ===
    genomad_output_dir = os.path.join(config['OUT_GENOMAD'], sample_id)
    db = config["GENOMAD_DB"]
    virsorter_output_dir = os.path.join(config["OUT_VIRSORT"], sample_id)
    dvf_output_dir = os.path.join(config["OUT_DVF"], sample_id)
    dvf_db = config["DVF_DB"]
    work_dir = config["WORK_DIR"]
    script_dir = config["SCRIPT_DIR"]
    viral_result = await(await viral_handle.run_tool(tool, unzipped_spades, genomad_output_dir, db, virsorter_output_dir, 
                                                      dvf_output_dir, dvf_db, work_dir, script_dir))

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
    print(quality_ratio)

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


# === Main function ===
async def main():
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor()
    ) as manager:
        viral_handle = await manager.launch(ViralDetectionAgent())
        checkv_handle = await manager.launch(CheckVAgent())
        cluster_handle = await manager.launch(DereplicationClusteringAgent())
        blast_handle = await manager.launch(BLASTAgent())

        # === Load configuration ===
        config_path = os.path.join(os.getcwd(), "config_py.sh")
        config = make_config(config_path)
        sample_ids_file = os.path.join(config['XFILE_DIR'], config['XFILE'])
        sample_ids = read_sample_ids(sample_ids_file)

        tool = random.choice(["GeNomad", "VirSorter2", "DeepVirFinder"])
        print(f"[MAIN] Chosen tool for all samples: {tool}")
        # === Run all samples in parallel ===
        first_sample_id = sample_ids[0]  # Choose one sample to return the derep_fasta

        per_sample_tasks = [
            asyncio.create_task(process_sample(sid, config, tool, viral_handle, checkv_handle, cluster_handle, first_sample_id))
            for sid in sample_ids
        ]

        results = await asyncio.gather(*per_sample_tasks)
        quality_ratios = [qr for _, qr in results if qr is not None]
        if quality_ratios:
            avg_quality_ratio = sum(quality_ratios) / len(quality_ratios)
            print(f"[MAIN] Average quality ratio: {avg_quality_ratio:.4f}")
        else:
            print("[MAIN] No quality ratios returned.")
        derep_fasta = next((df for df, _ in results if df is not None), None)
        if derep_fasta is None:
            raise ValueError("Dereplicated FASTA not found from any sample.")

        # === Cluster ===
        out_cluster = config["OUT_CLUSTER"]
        work_dir = config["WORK_DIR"]
        cluster_res_cluster = os.path.join(out_cluster, "clusterRes")
        tmp_dir_cluster = os.path.join(out_cluster, "tmp")
        rep_seq_src = os.path.join(out_cluster, "clusterRes_rep_seq.fasta")
        rep_seq_dst = os.path.join(work_dir, "query")
        out_derep = config["OUT_DEREP"]
        query_dir, cluster_file = await(await cluster_handle.run_cluster(
            sample_ids, out_derep, derep_fasta, out_cluster,
            cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst))

        # === BLAST ===
        db_dir = config["DB_DIR"]
        max_db_size = config["MAX_DB_SIZE"]
        db_list_path = os.path.join(db_dir, "db-list")
        prog = "05B_launchblast"
        fasta_dir = config["FASTA_DIR"]
        split_size = config["FA_SPLIT_FILE_SIZE"]
        results_dir = os.path.join(work_dir, "results_testing", prog)
        files_list_path = os.path.join(fasta_dir, "fasta-files")
        blast_results_dir = os.path.join(work_dir, "results_testing", "05C_blast")
        blast_type = config["BLAST_TYPE"]
        eval_param = config["EVAL"]
        out_fmt = config["OUT_FMT"]
        max_target_seqs = config["MAX_TARGET_SEQS"]
        merge_results_dir = os.path.join(work_dir, "results_testing", "05D_mergeblast")
        hits_file, match_ratio = await(await blast_handle.run_full_blast(
            work_dir, split_size, results_dir, query_dir, cluster_file, db_dir,
            blast_results_dir, blast_type, eval_param, out_fmt,
            max_target_seqs, merge_results_dir, max_db_size, db_list_path))
        print("BLAST Match Ratio: ", match_ratio)

        # === Annotation ===
        annotations_dir = config['ANNOTATIONS']
        out_annotate = config['OUTPUT']
        script_path = os.path.join(config['SCRIPT_DIR'], "solution1_manual.py")
        pctid = config['PCTID']
        length = config['LENGTH']
        final = annotate_blast(hits_file, annotations_dir, out_annotate, script_path, pctid, length)
        print(final)

if __name__ == "__main__":
    asyncio.run(main())


