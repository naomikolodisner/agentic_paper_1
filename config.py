from pathlib import Path
import os
import shutil

############################
# Base directories
############################

BASE = Path("/rs1/researchers/b/blhurwit/users")
PROJECT_ROOT = BASE / "nkolodi" / "agentic_paper_1"
RESULTS_ROOT = PROJECT_ROOT / "results"
DB_ROOT = BASE / "nkolodi" / "databases"
TOOL_ROOT = BASE / "nkolodi" / "tools"

############################
# Logs & scripts
############################

WORK_DIR = PROJECT_ROOT
LOG_DIR = PROJECT_ROOT / "logs"

############################
# Assembly inputs
############################

XFILE = "xad"
XFILE_DIR = PROJECT_ROOT / "data" / "sample_lists"
BACKGROUND_ASSEMBLIES = PROJECT_ROOT / "data" / "archive" / "background_assemblies"
SPIKE_IN_DIR = PROJECT_ROOT / "data" / "spike_in_samples"

############################
# InSilicoSeq (ISS) synthetic sample creation
############################

ISS_BIN = BASE / "nkolodi" / "conda_envs" / "insilicoseq" / "bin" / "iss"

# IMPORTANT: ISS's --draft treats each FILE passed to it as one genome (a
# single multi-record FASTA of many genomes concatenated together would be
# wrongly treated as ONE organism) -- verified empirically 2026-07-07. Any
# background-generation script must pass a directory of one-file-per-genome,
# globbed and passed as multiple --draft arguments.
HUMGUT_MAG_GLOB = "*.fasta"

# Spike-in grid: random subset of viral genomes per sample, spiked at a total
# read-fraction percentage. No ISS abundance distribution is used for
# spike-in -- read_mixing.sh uses ART Illumina instead, splitting the
# percentage-derived total viral read budget unevenly across the subset's
# genomes via ART_RATIO_WEIGHTS (see below), reusing the old
# equal2/equal3/equal4/unequal2/unequal3 scripts' fixed coverage ratios as
# relative weights instead of absolute coverages. read_mixing.sh loads
# SPIKE_PERCENTAGES directly (single source of truth) rather than hardcoding
# its own copy, and builds every percentage against every ART-compatible
# background row (exhaustive, not a random per-combo draw) -- so each sample
# dir gets len(SPIKE_PERCENTAGES) x (usable background rows) output combos.
#
# VIRAL_SUBSET_SIZES was originally [5, 10, 20] -- a small controlled
# diversity-scaling grid, unrelated to real viral community size. Replaced
# with realistic values after dereplicating/clustering the actual viral
# contigs found in the FMT samples (~80-150 distinct viruses per sample,
# down from ~900-1700 raw contigs pre-clustering). The INPHARED pool was
# expanded from 103 to 408 accessions (see INPHARED_ACCESSIONS_LIST above)
# specifically so subsets this large still draw from well under the full
# pool, keeping replicate draws independent.
VIRAL_SUBSET_SIZES = [80, 115, 150]
SPIKE_PERCENTAGES = [0.01, 0.05, 0.10]
NUM_REPLICATES_PER_SUBSET_SIZE = 1

############################
# ART Illumina (spike-in viral read simulation)
############################

ART_BIN = BASE / "nkolodi" / "conda_envs" / "test_env" / "bin" / "art_illumina"
ART_FRAGMENT_MEAN = 200
ART_FRAGMENT_SD = 10

# Cycled across a sample's genome subset in order (genome 1 -> 10, genome 2 ->
# 1, genome 3 -> 0.5, genome 4 -> 0.1, genome 5 -> 10 again, ...) -- the same
# absolute coverage values scripts/archive/create_single_equal_samples.sh and
# scripts/archive/create_unequal_samples_slurm.sh used directly, reused here
# as relative weights that split the pct-derived total viral read budget
# unevenly instead of equally.
ART_RATIO_WEIGHTS = [10, 1, 0.5, 0.1]

# ART has no native NovaSeq quality profile (confirmed via `art_illumina
# --help`); MSv3 is used as the closest available fallback -- ISS's novaseq
# model measures 151bp reads (verified by loading the model directly), which
# is 1bp over HS25's 150bp cap but comfortably under MSv3's 250bp one.
# Profile max supported read lengths: HS25 <=150bp, MSv3 <=250bp, NS50
# <=75bp -- if a background model's actual ISS-measured read length exceeds
# its mapped profile's cap, read_mixing.sh filters out that background row
# rather than invoking ART with an incompatible length.
ART_PROFILE_BY_MODEL = {
    "hiseq": "HS25",
    "miseq": "MSv3",
    "nextseq": "NS50",
    "novaseq": "MSv3",
}
ART_PROFILE_MAX_LEN = {
    "HS25": 150,
    "MSv3": 250,
    "NS50": 75,
}

############################
# Viral detection tools
############################

OUT_CHECKV = RESULTS_ROOT / "02_checkv"
CHECKVDB = DB_ROOT / "checkv-db-v1.5"
CHECKV_PARSER = PROJECT_ROOT / "scripts" / "CheckV_parser.R"
PARSE_LENGTH = 5000

# VirSorter2
OUT_VIRSORTER2 = RESULTS_ROOT / "01_viral_detection" / "01A_virsorter2"
OUT_CHECKV_VIRSORTER2 = OUT_CHECKV / "02A_virsorter2"

# DeepVirFinder
OUT_DVF = RESULTS_ROOT / "01_viral_detection" / "01B_dvf"
DVF_DB = DB_ROOT / "DeepVirFinder"
DVF_SCRIPT_DIR = BASE / "nkolodi" / "fmt_viruses" / "06_get_viruses"
OUT_CHECKV_DVF = OUT_CHECKV / "02B_dvf"

# geNomad
OUT_GENOMAD = RESULTS_ROOT / "01_viral_detection" / "01C_genomad"
GENOMAD_DB = DB_ROOT / "genomad_db"
OUT_CHECKV_GENOMAD = OUT_CHECKV / "02C_genomad"

# MARVEL
OUT_MARVEL = RESULTS_ROOT / "01_viral_detection" / "01D_marvel"
MARVEL_DB = DB_ROOT / "MARVEL"
OUT_CHECKV_MARVEL = OUT_CHECKV / "02D_marvel"

# VirFinder
OUT_VIRFINDER = RESULTS_ROOT / "01_viral_detection" / "01E_virfinder"
OUT_CHECKV_VIRFINDER = OUT_CHECKV / "02E_virfinder"

# VIBRANT
OUT_VIBRANT = RESULTS_ROOT / "01_viral_detection" / "01F_vibrant"
VIBRANT_DB = DB_ROOT / "VIBRANT" 
OUT_CHECKV_VIBRANT = OUT_CHECKV / "02F_vibrant"

# viralVerify
OUT_VIRALVERIFY = RESULTS_ROOT / "01_viral_detection" / "01G_viralverify"
HMM_DB = TOOL_ROOT / "viralVerify" 
OUT_CHECKV_VIRALVERIFY = OUT_CHECKV / "02G_viralverify"

# ViraMiner
OUT_VIRAMINER = RESULTS_ROOT / "01_viral_detection" / "01H_viraminer"
VIRAMINER_DB = TOOL_ROOT / "ViraMiner" 
OUT_CHECKV_VIRAMINER = OUT_CHECKV / "02H_viraminer"

# MetaPhinder 
OUT_METAPHINDER = RESULTS_ROOT / "01_viral_detection" / "01I_metaphinder"
BLAST_PATH = os.environ.get('BLAST_BIN_PATH', str(BASE / "nkolodi" / "conda_envs" / "blast" / "bin") + "/")
METAPHINDER_DB = TOOL_ROOT / "MetaPhinder"
OUT_CHECKV_METAPHINDER = OUT_CHECKV / "02I_metaphinder"

# Seeker 
OUT_SEEKER = RESULTS_ROOT / "01_viral_detection" / "01J_seeker"
OUT_CHECKV_SEEKER = OUT_CHECKV / "02J_seeker"

# VirSorter
OUT_VIRSORTER = RESULTS_ROOT / "01_viral_detection" / "01K_virsorter"
OUT_CHECKV_VIRSORTER = OUT_CHECKV / "02K_virsorter"
VIRSORTER_DB = DB_ROOT / "virsorter-data"
VIRSORTER_SCRIPT = TOOL_ROOT / "VirSorter" / "wrapper_phage_contigs_sorter_iPlant.pl" 

############################
# Dereplication & clustering
############################

OUT_DEREP = RESULTS_ROOT / "03_dereplicate"
OUT_CLUSTER = RESULTS_ROOT / "04_cluster"

############################
# BLAST setup
############################

DB_DIR = DB_ROOT / "AVrC"
MAX_DB_SIZE = "0.5GB"
# No longer used by pick_refs.py (superseded by INPHARED_* below) -- kept
# since DB_DIR/ANNOTATIONS (same AVrC database) still back the BLAST
# annotation step in pipeline/coordinator.py.
AVRC_ALL_SEQUENCES = DB_DIR / "AVrC_allsequences.fasta"

FASTA_DIR = PROJECT_ROOT / "query"
FA_SPLIT_FILE_SIZE = 5_000_000  # bytes

# BLAST parameters
BLAST_TYPE = "blastn"
MAX_TARGET_SEQS = 1
EVAL = 1e-3
OUT_FMT = 6

############################
# Annotation parameters
############################

PCTID = 85
LENGTH = 1000
BLAST_HITS = (
    RESULTS_ROOT
    / "05D_mergeblast"
    / "AVrC_allrepresentatives.fasta"
    / "clusterRes_rep_seq.fasta.txt"
)
ANNOTATIONS = DB_ROOT / "AVrC" / "database_csv"
OUTPUT = RESULTS_ROOT / "06_annotate"
# No longer used by pick_refs.py (see INPHARED_ACCESSIONS_LIST below) -- kept
# for the BLAST annotation step, which is unrelated to spike-in genome choice.
AVRC_METADATA_CSV = ANNOTATIONS / "AvRCv1.Merged_ViralDesc.csv"

############################
# INPHARED spike-in genome catalog
############################

# pick_refs.py's spike-in genome pool: every accession in the current full
# INPHARED catalog (~2902 genomes, /rs1/shares/brc/databases/inphared/
# all_genomes.txt) confirmed present in kraken2_pluspfp's seqid2taxid.map --
# written by scripts/check_kraken2_accessions.py (408 as of the last run;
# supersedes the smaller 103-accession VirFinder/INPHARED pre-2014-overlap
# pool and its archived scripts/archive/check_kraken2_accessions.sh). Each
# accession is its own single-record FASTA file in INPHARED_GENOMES_DIR.
INPHARED_ACCESSIONS_LIST = PROJECT_ROOT / "data" / "accessions_in_kraken2.txt"
INPHARED_GENOMES_DIR = Path("/rs1/shares/brc/databases/inphared/GenomesDB")

############################
# Archived (superseded, not read by the active pipeline)
############################

# Old tool-benchmarking ground truth (Set3 simulated metagenomes). No longer
# used for anything -- the new FMT-derived datasets built by the current
# pipeline are the ground truth going forward. Kept pointed at its actual
# archived location (data/archive/set3_simulated_metagenomes/) rather than
# deleted, per this project's convention of retaining superseded artifacts.
SPADES_DIR = PROJECT_ROOT / "data" / "archive" / "set3_simulated_metagenomes" / "assemblies"
CONTIG_IDENTITIES = PROJECT_ROOT / "data" / "archive" / "set3_simulated_metagenomes" / "contig_identities.csv"

# Old raw-HumGut-subset background generation (scripts/archive/generate_background.sh
# + scripts/archive/extract_humgut_subset.py), driven by a built-in ISS
# abundance distribution (lognormal/exponential) rather than a real profile.
# Superseded by the FMT-profile-driven background (see the active
# "FMT-derived background" config section). Output already moved to
# data/archive/background_iss/.
HUMGUT_PROPHAGE_REMOVED_DIR = DB_ROOT / "HumGut" / "prophage_removed"
HUMGUT_TAR = Path("/rs1/shares/brc/admin/databases/HumGut/HumGut2.tar")
HUMGUT_TSV = Path("/rs1/shares/brc/admin/databases/HumGut/HumGut2.tsv")
HUMGUT_SUBSET_N = 500
HUMGUT_SUBSET_SEED = 42
HUMGUT_RAW_MAGS_DIR = DB_ROOT / "HumGut" / "raw_subset"
BACKGROUND_ISS_DIR = PROJECT_ROOT / "data" / "archive" / "background_iss"
BACKGROUND_MANIFEST = BACKGROUND_ISS_DIR / "manifest.tsv"
ISS_MODELS = ["hiseq", "novaseq"]
BACKGROUND_ABUNDANCE_DISTS = ["lognormal", "exponential"]
BACKGROUND_N_READS = ["1M"]  # 1M total reads (R1+R2) = 500k read pairs

############################
# Kraken2 (check gate)
############################

KRAKEN2_DB = Path("/rs1/shares/brc/admin/databases/kraken2_pluspfp")
KRAKEN2_SEQID2TAXID = KRAKEN2_DB / "seqid2taxid.map"

############################
# Utility functions
############################

def init_dir(*dirs: Path):
    """
    Create directories if missing; if they exist, empty them.
    """
    for d in dirs:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def create_dir(*dirs: Path):
    """
    Create directories if they do not exist.
    """
    for d in dirs:
        if not d.exists():
            print(f"{d} does not exist. Directory created")
            d.mkdir(parents=True, exist_ok=True)


def lc(file: Path) -> int:
    """
    Line count (wc -l equivalent).
    """
    with file.open() as f:
        return sum(1 for _ in f)


