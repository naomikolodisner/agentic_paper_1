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
SPADES_DIR = PROJECT_ROOT / "data" / "set3_simulated_metagenomes" / "assemblies"
BACKGROUND_ASSEMBLIES = PROJECT_ROOT / "data" / "background_assemblies"
CONTIG_IDENTITIES = PROJECT_ROOT / "data" / "set3_simulated_metagenomes" / "contig_identities.csv"
SPIKE_IN_DIR = PROJECT_ROOT / "data" / "spike_in_samples"

############################
# InSilicoSeq (ISS) synthetic sample creation
############################

ISS_BIN = BASE / "nkolodi" / "conda_envs" / "insilicoseq" / "bin" / "iss"

# Directory of PHORAGER's prophage-stripped HumGut MAGs -- one FASTA file per
# MAG (each may be multi-contig). PHORAGER has not been run on HumGut yet
# (HumGut is currently just the downloaded tar at
# /rs1/shares/brc/admin/databases/HumGut/HumGut2.tar; PHORAGER itself lives at
# /rs1/researchers/b/blhurwit/work/PHORAGER). Update HUMGUT_RAW_MAGS_DIR usage
# in generate_background.sh to this dir once that run completes -- for now
# background is generated from raw (prophage-intact) HumGut MAGs instead.
#
# IMPORTANT: ISS's --draft treats each FILE passed to it as one genome (a
# single multi-record FASTA of many MAGs concatenated together would be
# wrongly treated as ONE organism) -- verified empirically 2026-07-07. This
# must stay a directory of one-file-per-MAG, globbed and passed as multiple
# --draft arguments by generate_background.sh.
HUMGUT_PROPHAGE_REMOVED_DIR = DB_ROOT / "HumGut" / "prophage_removed"
HUMGUT_MAG_GLOB = "*.fasta"

# Raw HumGut MAG source (prophages intact) -- extracted by
# scripts/extract_humgut_subset.py from the tarball below into one plain
# FASTA per genome in HUMGUT_RAW_MAGS_DIR, since HumGut2.tar has never been
# extracted (only the tar + its manifest TSV exist on disk). HumGut2.tsv has
# ~31,226 genome rows total; HUMGUT_SUBSET_N controls how many of those are
# extracted for use as background draft genomes (fixed seed => reproducible).
HUMGUT_TAR = Path("/rs1/shares/brc/admin/databases/HumGut/HumGut2.tar")
HUMGUT_TSV = Path("/rs1/shares/brc/admin/databases/HumGut/HumGut2.tsv")
HUMGUT_SUBSET_N = 500
HUMGUT_SUBSET_SEED = 42
HUMGUT_RAW_MAGS_DIR = DB_ROOT / "HumGut" / "raw_subset"

BACKGROUND_ISS_DIR = PROJECT_ROOT / "data" / "background_iss"
BACKGROUND_MANIFEST = BACKGROUND_ISS_DIR / "manifest.tsv"

# Only hiseq and novaseq are usable for the ART-driven spike-in step: ISS's
# miseq/nextseq models both measure ~301bp reads, which exceed every ART
# built-in profile (largest is MSv3 at 250bp) -- see ART_PROFILE_BY_MODEL /
# ART_PROFILE_MAX_LEN below. generate_background.sh loads this list directly
# (single source of truth) rather than hardcoding its own MODELS array.
ISS_MODELS = ["hiseq", "novaseq"]

# Background grid: 2 distributions (a "typical" lognormal community and a
# skewed "abnormal" one -- a few genomes very abundant, most very low) x 1
# read depth. BACKGROUND_N_READS is deliberately a list (not a scalar) so
# adding a shallower depth later (e.g. "0.1M") just grows the grid -- the
# nested-loop combo builder in generate_background.sh doesn't change.
BACKGROUND_ABUNDANCE_DISTS = ["lognormal", "exponential"]
BACKGROUND_N_READS = ["1M"]  # 1M total reads (R1+R2) = 500k read pairs

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
VIRAL_SUBSET_SIZES = [5, 10, 20]
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

# pick_refs.py's spike-in genome pool: the 103 accessions from the
# VirFinder/INPHARED pre-2014 overlap list (106 total,
# in_inphared_and_virfinder_pre2014) confirmed present in kraken2_pluspfp's
# seqid2taxid.map -- written by the archived
# scripts/archive/check_kraken2_accessions.sh. Each accession is its own
# single-record FASTA file in INPHARED_GENOMES_DIR.
INPHARED_ACCESSIONS_LIST = PROJECT_ROOT / "data" / "accessions_in_kraken2.txt"
INPHARED_GENOMES_DIR = Path(
    "/rs1/shares/brc/databases/inphared/genomes_in_inphared_and_virfinder_pre2014"
)

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


