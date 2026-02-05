from pathlib import Path
import shutil

############################
# Base directories
############################

XDISK = Path("/xdisk/gwatts")
PROJECT_ROOT = XDISK / "kolodisner" / "agentic_paper_1"
RESULTS_ROOT = PROJECT_ROOT / "results"
DB_ROOT = Path("/groups/gwatts/databases")

############################
# Logs & scripts
############################

WORK_DIR = PROJECT_ROOT
LOG_DIR = PROJECT_ROOT / "logs"

############################
# Assembly inputs
############################

XFILE = "xac"
XFILE2 = "xad"
XFILE_DIR = PROJECT_ROOT
SPADES_DIR = PROJECT_ROOT / "out_spades"

############################
# Viral detection tools
############################

OUT_CHECKV = RESULTS_ROOT / "02_checkv"
CHECKVDB = DB_ROOT / "checkv-db-v1.5"
CHECKV_PARSER = PROJECT_ROOT / "CheckV_parser.R"
PARSE_LENGTH = 5000

# VirSorter2
OUT_VIRSORT = RESULTS_ROOT / "01_viral_detection" / "01A_virsorter2"
OUT_CHECKV_VIRSORT = OUT_CHECKV / "02A_virsorter2"

# DeepVirFinder
OUT_DVF = RESULTS_ROOT / "01_viral_detection" / "01B_dvf"
DVF_DB = DB_ROOT / "DeepVirFinder"
OUT_CHECKV_DVF = OUT_CHECKV / "02B_dvf"

# geNomad
OUT_GENOMAD = RESULTS_ROOT / "01_viral_detection" / "01C_genomad"
GENOMAD_DB = DB_ROOT / "genomad_db"
OUT_CHECKV_GENOMAD = OUT_CHECKV / "02C_genomad"

# MARVEL
OUT_MARVEL = RESULTS_ROOT / "01_viral_detection" / "01D_marvel"
MARVEL_DB = DB_ROOT / "MARVEL"
OUT_CHECKV_MARVEL = OUT_CHECKV / "02D_marvel"

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


