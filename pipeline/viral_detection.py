import asyncio
import parsl
from parsl import python_app
from academy.agent import Agent, action

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# Parsl apps — each activates its own conda environment.
# All apps are pinned to the 'viral_htex' executor defined in combined_config,
# which is loaded once in coordinator.py main() before any agents are created.
# --------------------------------------------------------------------------- #

@python_app(executors=['viral_htex'])
def unzip_fasta_app(spades_gz, unzipped_spades_path):
    import subprocess
    import os
    import socket
    print("Unzip Running on node:", socket.gethostname(), flush=True)
    if os.path.exists(unzipped_spades_path):
        return unzipped_spades_path
    try:
        subprocess.run(["gzip", "-dk", spades_gz], check=True)
    except subprocess.CalledProcessError as e:
        raise e
    return unzipped_spades_path


@python_app(executors=['viral_htex'])
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
        "conda", "run", "-n", "virsorter2",
        "virsorter", "run", "-w", virsorter2_output_dir,
        "-i", unzipped_spades, "--min-length", "1500", "-j", "4", "all",
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(virsorter2_output_dir, "final-viral-combined.fa")


@python_app(executors=['viral_htex'])
def deepvirfinder_app(unzipped_spades, dvf_output_dir, dvf_db, work_dir, script_dir):
    import subprocess
    import os
    import socket
    print("DeepVirFinder Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(dvf_output_dir):
        os.makedirs(dvf_output_dir)
    cmd = [
        "conda", "run", "-n", "dvf",
        "python", "dvf.py",
        "-i", unzipped_spades,
        "-o", dvf_output_dir,
        "-l", "1500",
    ]
    # cwd=dvf_db: dvf.py must be invoked from its own directory (relative import paths)
    subprocess.run(cmd, check=True, cwd=dvf_db)
    dvf_output = os.path.join(
        dvf_output_dir,
        os.path.basename(unzipped_spades) + "_gt1500bp_dvfpred.txt",
    )
    dvf_fasta_output = os.path.join(dvf_output_dir, "dvf.fasta")
    cmd2 = [
        "conda", "run", "-n", "dvf",
        "python", "id_from_fasta.py",
        "-c", unzipped_spades,
        "-d", dvf_output,
        "-o", dvf_fasta_output,
    ]
    subprocess.run(cmd2, check=True, cwd=script_dir)
    return dvf_fasta_output


@python_app(executors=['viral_htex'])
def genomad_app(unzipped_spades, genomad_output_dir, genomad_db):
    import subprocess
    import os
    import socket
    print("GeNomad Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(genomad_output_dir):
        os.makedirs(genomad_output_dir)
    cmd = [
        "conda", "run", "-n", "genomad",
        "genomad", "end-to-end", "--cleanup", "--restart",
        unzipped_spades, genomad_output_dir, genomad_db,
    ]
    subprocess.run(cmd, check=True)
    # geNomad names output dirs/files after the input FASTA stem, not "contigs"
    stem = os.path.splitext(os.path.basename(unzipped_spades))[0]
    return os.path.join(genomad_output_dir, stem + "_summary", stem + "_virus.fna")


@python_app(executors=['viral_htex'])
def marvel_app(unzipped_spades, marvel_output_dir, marvel_db):
    import subprocess
    import os
    import socket
    print("MARVEL Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(marvel_output_dir):
        os.makedirs(marvel_output_dir)
    unzipped_spades_marvel = os.path.dirname(unzipped_spades)
    cmd = [
        "conda", "run", "-n", "marvel", "python3",
        "marvel_bins.py", "-i", unzipped_spades_marvel, "-t", "16",
        "-o", marvel_output_dir,
    ]
    subprocess.run(cmd, cwd=marvel_db, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return os.path.join(marvel_output_dir, "prokka", "contigs", "prokka_results_contigs.fna")


@python_app(executors=['viral_htex'])
def virfinder_app(unzipped_spades, virfinder_output_dir):
    import subprocess
    import os
    import socket
    print("VirFinder Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(virfinder_output_dir):
        os.makedirs(virfinder_output_dir)
    vf_tsv = os.path.join(virfinder_output_dir, "virfinder_results.tsv")
    viral_fasta = os.path.join(virfinder_output_dir, "viral_contigs.fna")
    ids_file = os.path.join(virfinder_output_dir, "viral_ids.txt")
    q_cutoff = 0.05
    r_script = f"""
    library(VirFinder)
    res <- VF.pred("{unzipped_spades}")
    write.table(res, file="{vf_tsv}", sep="\\t", row.names=FALSE, quote=FALSE)
    """
    subprocess.run(
        ["conda", "run", "-n", "virfinder", "Rscript", "-e", r_script],
        check=True,
    )
    cmd_filter = (
        f"awk -F '\\t' 'NR>1 && $5 < {q_cutoff} {{print $1}}' "
        f"{vf_tsv} > {ids_file}"
    )
    subprocess.run(cmd_filter, shell=True, check=True)
    cmd_extract = [
        "conda", "run", "-n", "seqtk", "seqkit", "grep",
        "-f", ids_file, unzipped_spades,
    ]
    with open(viral_fasta, "w") as out_f:
        subprocess.run(cmd_extract, check=True, stdout=out_f)
    return viral_fasta


@python_app(executors=['viral_htex'])
def vibrant_app(unzipped_spades, vibrant_db, vibrant_output_dir):
    import subprocess
    import os
    import socket
    print("VIBRANT Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(vibrant_output_dir):
        os.makedirs(vibrant_output_dir)
    vibrant_script = os.path.join(vibrant_db, "VIBRANT/VIBRANT_run.py")
    cmd = [
        "conda", "run", "-n", "vibrant",
        "python3", vibrant_script,
        "-i", unzipped_spades,
        "-folder", vibrant_output_dir, "-d", vibrant_db,
        "-t", "16", "-f", "nucl", "-no_plot",
    ]
    subprocess.run(cmd, check=True)
    return vibrant_output_dir  # directory; caller must locate the viral FASTA inside


@python_app(executors=['viral_htex'])
def viralverify_app(unzipped_spades, viralverify_output_dir, hmm_db):
    import subprocess
    import os
    import socket
    print("viralVerify Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(viralverify_output_dir):
        os.makedirs(viralverify_output_dir)
    viralverify = os.path.join(hmm_db, "bin", "viralverify")
    viralverify_db = os.path.join(hmm_db, "nbc_hmms.hmm")
    cmd = [
        "conda", "run", "-n", "viralverify", viralverify, "-f",
        unzipped_spades, "-o", viralverify_output_dir, "--hmm", viralverify_db,
    ]
    subprocess.run(cmd, check=True)
    return viralverify_output_dir  # directory; caller must locate the viral FASTA inside


@python_app(executors=['viral_htex'])
def viraminer_app(unzipped_spades, viraminer_db, viraminer_output_dir):
    import subprocess
    import os
    import socket
    import csv
    from Bio import SeqIO
    print("ViraMiner Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(viraminer_output_dir):
        os.makedirs(viraminer_output_dir)
    csv_file = os.path.join(viraminer_output_dir, "viraminer_input.csv")
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for record in SeqIO.parse(unzipped_spades, "fasta"):
            seq = str(record.seq).strip()
            for i in range(0, len(seq), 300):
                chunk = seq[i:i + 300]
                if len(chunk) == 300:
                    writer.writerow([record.id + "_chunk" + str(i), chunk, 0])
        writer.writerow(["dummy_seq", "A" * 300, 1])
    output_txt = os.path.join(viraminer_output_dir, "viraminer_predictions.txt")
    viraminer_script = os.path.join(viraminer_db, "predict_only.py")
    viraminer_model = os.path.join(
        viraminer_db, "final_ViraMiner", "final_ViraMiner_beforeFT.hdf5"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "viraminer", "python", viraminer_script,
        "--input_file", csv_file, "--model_path", viraminer_model,
    ]
    with open(output_txt, "w") as f:
        subprocess.run(cmd, stdout=f, check=True, env=env)
    return output_txt  # predictions TXT, not FASTA; F1 evaluation skipped for this tool


@python_app(executors=['viral_htex'])
def metaphinder_app(unzipped_spades, metaphinder_db, blast_path, metaphinder_output_dir):
    import subprocess
    import os
    import socket
    print("MetaPhinder Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(metaphinder_output_dir):
        os.makedirs(metaphinder_output_dir)
    metaphinder_script = os.path.join(metaphinder_db, "MetaPhinder.py")
    metaphinder_data = os.path.join(metaphinder_db, "database", "ALL_140821_hr")
    cmd = [
        "conda", "run", "-n", "metaphinder", "python", metaphinder_script,
        "-i", unzipped_spades, "-d", metaphinder_data, "-b", blast_path,
        "-o", metaphinder_output_dir,
    ]
    subprocess.run(cmd, cwd=metaphinder_db, check=True)
    return os.path.join(metaphinder_output_dir, "output.txt")


@python_app(executors=['viral_htex'])
def seeker_app(unzipped_spades, seeker_output_dir):
    import subprocess
    import os
    import socket
    import textwrap
    print("Seeker Running on node:", socket.gethostname(), flush=True)
    if not os.path.exists(seeker_output_dir):
        os.makedirs(seeker_output_dir)
    threshold = 0.5
    out_fasta = os.path.join(seeker_output_dir, "seeker_phage_contigs.fa")
    seeker_script = textwrap.dedent(f"""
        from seeker import SeekerFasta
        seeker_fasta = SeekerFasta("{unzipped_spades}")
        seeker_fasta.meta2fasta(out_fasta_path="{out_fasta}", threshold={threshold})
    """)
    subprocess.run(["conda", "run", "-n", "seeker", "python", "-c", seeker_script], check=True)
    return out_fasta


@python_app(executors=['viral_htex'])
def virsorter_app(unzipped_spades, virsorter_db, virsorter_script, virsorter_output_dir):
    import subprocess
    import os
    import shutil
    import socket
    print("VirSorter Running on node:", socket.gethostname(), flush=True)
    if os.path.exists(virsorter_output_dir):
        shutil.rmtree(virsorter_output_dir)
    os.makedirs(virsorter_output_dir)
    cmd = [
        "conda", "run", "-n", "virsorter", "perl", virsorter_script,
        "-f", unzipped_spades, "--db", "1", "--wdir", virsorter_output_dir,
        "--ncpu", "4", "--data-dir", virsorter_db,
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(
        virsorter_output_dir,
        "Predicted_viral_sequences",
        "VIRSorter_predicted_viral_sequences.fasta",
    )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class ViralDetectionAgent(Agent):
    def __init__(self):
        super().__init__()

    @action
    async def run_tool(
        self, tool, unzipped_spades, genomad_output_dir, genomad_db,
        virsorter2_output_dir, dvf_output_dir, dvf_db, work_dir, script_dir,
        marvel_output_dir, marvel_db, virfinder_output_dir, vibrant_db,
        vibrant_output_dir, viralverify_output_dir, hmm_db, viraminer_db,
        viraminer_output_dir, metaphinder_db, blast_path, metaphinder_output_dir,
        seeker_output_dir, virsorter_db, virsorter_script, virsorter_output_dir,
    ) -> str:
        dispatch = {
            "GeNomad":      lambda: genomad_app(unzipped_spades, genomad_output_dir, genomad_db),
            "VirSorter2":   lambda: virsorter2_app(unzipped_spades, virsorter2_output_dir),
            "DeepVirFinder":lambda: deepvirfinder_app(unzipped_spades, dvf_output_dir, dvf_db, work_dir, script_dir),
            "MARVEL":       lambda: marvel_app(unzipped_spades, marvel_output_dir, marvel_db),
            "VirFinder":    lambda: virfinder_app(unzipped_spades, virfinder_output_dir),
            "VIBRANT":      lambda: vibrant_app(unzipped_spades, vibrant_db, vibrant_output_dir),
            "viralVerify":  lambda: viralverify_app(unzipped_spades, viralverify_output_dir, hmm_db),
            "ViraMiner":    lambda: viraminer_app(unzipped_spades, viraminer_db, viraminer_output_dir),
            "MetaPhinder":  lambda: metaphinder_app(unzipped_spades, metaphinder_db, blast_path, metaphinder_output_dir),
            "Seeker":       lambda: seeker_app(unzipped_spades, seeker_output_dir),
            "VirSorter":    lambda: virsorter_app(unzipped_spades, virsorter_db, virsorter_script, virsorter_output_dir),
        }
        future = dispatch[tool]()
        return await asyncio.to_thread(future.result)
