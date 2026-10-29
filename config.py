from pathlib import Path
import os
import shutil

############################
# Base directories
############################

BASE = Path("/rs1/researchers/b/blhurwit/users")
PROJECT_ROOT = BASE / "nkolodi" / "agentic_paper_1"
RESULTS_ROOT = PROJECT_ROOT / "agentic_benchmarking" / "results"
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

# Spike-in grid: random subset of viral genomes per sample, each genome
# spiked in at the SAME fixed ART fold coverage ("plain" spike-in, like the
# original mock_datasets/archive/create_single_equal_samples.sh equal-coverage
# design) -- one SPIKE_COVERAGES value per output combo, applied identically
# to every genome in the subset. 10_read_mixing.sh loads SPIKE_COVERAGES
# directly (single source of truth) and builds every coverage against every
# ART-compatible background row (exhaustive), so each sample dir gets
# len(SPIKE_COVERAGES) x (usable background rows) output combos. The
# resulting viral read percentage is a DERIVED, logged quantity
# (spike_summary.tsv per combo), never an input.
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
# Original equal-coverage grid (create_single_equal_samples.sh). 0.1x/0.5x
# are below assemblable depth -- kept deliberately as detection-limit levels.
SPIKE_COVERAGES = [0.1, 0.5, 1, 10]

# "OG" spike-in design -- an exact replica of the PI's original ART scripts
# (mock_datasets/archive/create_single_equal_samples.sh,
# create_unequal_samples_slurm.sh; output in data/archive/{single,equal2,...}/),
# but with genomes drawn fresh from the INPHARED pool and the FMT background.
# Each type: n_viruses genomes per sample, SPIKE_OG_REPLICATES samples, and a
# set of named conditions, each a list of per-genome ART -f coverages (in
# contigs.fasta order). 09_pick_refs.py --design og writes these to each sample
# dir's conditions.tsv; 10_read_mixing.sh builds one output dir per condition.
SPIKE_OG_REPLICATES = 1  # PI's original used 15; 1 for initial testing
_EQUAL_CONDITIONS = lambda n: {f"{c:g}x": [c] * n for c in SPIKE_COVERAGES}
SPIKE_OG_DESIGNS = {
    "single":   {"n_viruses": 1, "conditions": _EQUAL_CONDITIONS(1)},
    "equal2":   {"n_viruses": 2, "conditions": _EQUAL_CONDITIONS(2)},
    "equal3":   {"n_viruses": 3, "conditions": _EQUAL_CONDITIONS(3)},
    "equal4":   {"n_viruses": 4, "conditions": _EQUAL_CONDITIONS(4)},
    "unequal2": {"n_viruses": 2, "conditions": {"10x_1x": [10, 1],
                                                "1x_0.5x": [1, 0.5]}},
    "unequal3": {"n_viruses": 3, "conditions": {"10x_1x_0.5x": [10, 1, 0.5],
                                                "1x_0.5x_0.1x": [1, 0.5, 0.1]}},
}
NUM_REPLICATES_PER_SUBSET_SIZE = 1

############################
# ART Illumina (spike-in viral read simulation)
############################

ART_BIN = BASE / "nkolodi" / "conda_envs" / "test_env" / "bin" / "art_illumina"
ART_FRAGMENT_MEAN = 200
ART_FRAGMENT_SD = 10

# ART has no native NovaSeq quality profile (confirmed via `art_illumina
# --help`); MSv3 is used as the closest available fallback -- ISS's novaseq
# model measures 151bp reads (verified by loading the model directly), which
# is 1bp over HS25's 150bp cap but comfortably under MSv3's 250bp one.
# Profile max supported read lengths: HS25 <=150bp, MSv3 <=250bp, NS50
# <=75bp -- if a background model's actual ISS-measured read length exceeds
# its mapped profile's cap, 10_read_mixing.sh filters out that background row
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
CHECKV_PARSER = PROJECT_ROOT / "agentic_benchmarking" / "scripts" / "CheckV_parser.R"
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
# No longer used by 09_pick_refs.py (superseded by INPHARED_* below) -- kept
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
OUTPUT = RESULTS_ROOT / "05_annotate"  # was 06_annotate -- renumbered after
                                       # kraken2_fmt/kraken2_check moved out
                                       # of results/ into mock_datasets/
# No longer used by 09_pick_refs.py (see INPHARED_ACCESSIONS_LIST below) -- kept
# for the BLAST annotation step, which is unrelated to spike-in genome choice.
AVRC_METADATA_CSV = ANNOTATIONS / "AvRCv1.Merged_ViralDesc.csv"

############################
# INPHARED spike-in genome catalog
############################

# 09_pick_refs.py's spike-in genome pool: every accession in the current full
# INPHARED catalog (~2902 genomes, /rs1/shares/brc/databases/inphared/
# all_genomes.txt) confirmed present in kraken2_pluspfp's seqid2taxid.map --
# written by mock_datasets/scripts/08_check_kraken2_accessions.py (408 as of the last run;
# supersedes the smaller 103-accession VirFinder/INPHARED pre-2014-overlap
# pool and its archived mock_datasets/archive/check_kraken2_accessions.sh). Each
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

# Old raw-HumGut-subset background generation (mock_datasets/archive/generate_background.sh
# + mock_datasets/archive/extract_humgut_subset.py), driven by a built-in ISS
# abundance distribution (lognormal/exponential) rather than a real profile.
# Superseded by the FMT-profile-driven background (see the active
# "FMT-derived background" config section). Output already moved to
# data/archive/background_iss/.
# HUMGUT_TSV and HUMGUT_PROPHAGE_REMOVED_DIR moved to the active section
# below 2026-09-21 -- same file/path as before, but now used for
# representative-genome selection and the real phoraged-genome extraction
# target, not just the archived raw-subset extraction.
HUMGUT_TAR = Path("/rs1/shares/brc/admin/databases/HumGut/HumGut2.tar")
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
# FMT-derived background: kraken2 classification -> ISS abundance profile
############################

# Custom HumGut kraken2 database (built 2026-09-08 from the raw HumGut
# genome collection, GTDB taxonomy; human library NOT included -- separate
# from KRAKEN2_DB/kraken2_pluspfp above, which is only used by the spike-in
# check gate). Built with the kraken2env conda env (kraken2 2.1.3) --
# classification must use the same env/version.
HUMGUT_KRAKEN2_DB = PROJECT_ROOT / "data" / "humgut_kraken2"

# Real FMT SPAdes assemblies (contigs.fasta per sample), produced upstream by
# the separate fmt_viruses repo -- not part of this project's own pipeline.
FMT_ASSEMBLIES_DIR = BASE / "nkolodi" / "fmt_viruses" / "out_spades"

# Pilot list of real FMT samples to classify against HUMGUT_KRAKEN2_DB, used
# to build the FMT-profile-driven background (replacing the old built-in ISS
# abundance distributions -- see BACKGROUND_ABUNDANCE_DISTS above, archived).
# SUPERSEDED 2026-09-19 by the read-based pass below (FMT_READS_*): ERR2198611-613
# only have a SPAdes assembly on Hazel (their raw reads lived on a now-
# inaccessible external /xdisk cluster), so this classified contigs.fasta
# weighted by SPAdes coverage as a depth proxy. Its script (now
# mock_datasets/archive/kraken2_fmt.sh, moved 2026-09-22) and output are kept --
# still valid, just not the sample set going forward.
FMT_SAMPLE_LIST = XFILE_DIR / "xac"

# Lives under mock_datasets/, not RESULTS_ROOT -- this is a dataset-prep
# output (feeds the FMT abundance profile), not an agentic-workflow stage.
OUT_KRAKEN2_FMT = PROJECT_ROOT / "mock_datasets" / "results" / "kraken2_fmt"

# Read-based pass (preferred going forward): ERR2198703-705 have their
# actual host(human)-decontaminated raw paired reads sitting on Hazel
# already (bowtie2 --un-conc-gz vs. CHM13, run by the upstream fmt_viruses
# repo -- see fmt_viruses/03_contam_removal/run_remove_human.sh), so these
# can be classified directly by kraken2 instead of going through an
# assembly-coverage proxy. Standard kraken2->abundance practice is to follow
# up with Bracken (redistributes reads whose LCA collapsed above genome
# level back down) rather than using the raw kraken2 --report counts
# directly -- not yet decided/built as of this comment.
# FMT_READS_DIR holds one dir per sample with r1.fastq.gz / r2.fastq.gz
# symlinks to FMT_READS_SOURCE_DIR's {sample}_1/_2.fastq.gz (same layout as
# the Nextflow kraken2 process expects; the originals are left untouched).
FMT_READS_SOURCE_DIR = BASE / "nkolodi" / "fmt_viruses" / "03_contam_removal"
FMT_READS_DIR = PROJECT_ROOT / "data" / "fmt_reads"
FMT_READS_SAMPLE_LIST = XFILE_DIR / "xae"
OUT_KRAKEN2_FMT_READS = PROJECT_ROOT / "mock_datasets" / "results" / "kraken2_fmt_reads"

# HumGut's own manifest -- our kraken2 DB's taxid IS HumGut's gtdb_tax_id
# column directly (verified 2026-09-20: 4940 distinct gtdb_tax_id values in
# the manifest vs. 4939 distinct genome taxids seen in the fixed Bracken
# index), which is coarser than HumGut's own genome-level cluster975
# grouping -- several cluster975 genome files can share one gtdb_tax_id.
# mock_datasets/scripts/04_pick_representative_genomes.py uses this to choose one genome_file
# per taxid (highest prevalence_score among that taxid's candidate rows --
# how consistently that specific genome was detected across HumGut's
# reference metagenomes, chosen over a dRep-style completeness/contamination
# quality score since the goal is a realistic background community, not the
# cleanest reference assembly).
HUMGUT_TSV = Path("/rs1/shares/brc/admin/databases/HumGut/HumGut2.tsv")
OUT_REPRESENTATIVE_GENOMES = PROJECT_ROOT / "mock_datasets" / "results" / "representative_genomes.tsv"

# mock_datasets/scripts/06_build_fmt_abundance_profile.py: one ISS-ready abundance file per
# sample (config.FMT_READS_SAMPLE_LIST), summing each representative
# genome's Bracken-corrected read count and normalizing to a fraction. The
# first column is the real, absolute, resolved path to that genome's
# extracted phoraged FASTA under HUMGUT_PROPHAGE_REMOVED_DIR (below) --
# verified in ISS's own source (iss/generator.py) that --abundance_file +
# --draft matches the abundance file's genome_id against the LITERAL file
# path string passed to --draft, not a basename or FASTA record ID, so
# whatever calls `iss generate --draft` should build its file list FROM this
# file's own first column (e.g. `cut -f1`) rather than reconstructing paths
# independently. Also filters out any representative genome PHORAGER never
# processed (534/31225 HumGut genomes were never phoraged at all -- see
# HUMGUT_PROPHAGE_REMOVED_DIR below; 14-22 per sample were affected) and
# renormalizes -- negligible abundance-mass impact, logged per sample.
OUT_FMT_ABUNDANCE_PROFILES = PROJECT_ROOT / "mock_datasets" / "results" / "fmt_abundance_profiles"

# mock_datasets/scripts/helpers/extract_humgut_phoraged.py: pulls only the ~3455 genomes our
# profiles actually need out of the shared HumGut_phoraged_*.tar.gz shards
# (PHORAGER's prophage-stripped HumGut genomes, /rs1/shares/brc/admin/
# databases/humgut_phoraged/ -- entire ~31225-genome collection across 7
# archives, each inner file individually gzipped). Archives key files by
# HumGut_name (e.g. "HumGut_2753"), joined through HUMGUT_TSV's
# HumGut_name/genome_file columns to our representative-genome table's
# genome_file identifiers. Output: one genome per "<genome_id>.fna" file
# (genome_id = the bare accession, matching OUT_FMT_ABUNDANCE_PROFILES'
# genome_id column) -- was HUMGUT_PROPHAGE_REMOVED_DIR under "Archived"
# above (raw-MAG-subset-era var, same path, now genuinely populated).
HUMGUT_PROPHAGE_REMOVED_DIR = DB_ROOT / "HumGut" / "prophage_removed"

# mock_datasets/scripts/07_generate_background_fmt.sh: runs `iss generate --abundance_file`
# per real FMT sample profile (config.OUT_FMT_ABUNDANCE_PROFILES) against
# the extracted phoraged genomes, replacing the archived lognormal/
# exponential grid (BACKGROUND_ABUNDANCE_DISTS et al. above) with one driven
# by real gut-microbiome abundance. Grid: FMT_BACKGROUND_MODELS x
# FMT_READS_SAMPLE_LIST x FMT_BACKGROUND_N_READS. Models restricted to
# hiseq/novaseq for the same reason as the old ISS_MODELS -- ART (which
# 10_read_mixing.sh uses for viral spike-in) has no built-in profile that fits
# miseq/nextseq's ~301bp read length. Manifest schema is a direct match for
# the one 10_read_mixing.sh already expects (BACKGROUND_MANIFEST above) --
# whatever repoints 10_read_mixing.sh at OUT_BACKGROUND_FMT should need no
# further changes to how it reads that file.
OUT_BACKGROUND_FMT = PROJECT_ROOT / "data" / "background_fmt"
BACKGROUND_FMT_MANIFEST = OUT_BACKGROUND_FMT / "manifest.tsv"
# novaseq-only, 10M total reads (5M pairs) -- per direction 2026-09-21.
# The old 1M-read default (from the archived 500-genome raw-MAG background)
# would badly undersample this community: our profiles have 2321-3115
# genomes with a realistic long-tailed abundance distribution (one genome
# ~21%, then a long tail), so 500k pairs would average ~150-215 pairs/genome
# and leave most of the tail at single-digit or zero reads. 5M pairs keeps
# even a ~0.01%-abundance genome at ~500 pairs -- enough to actually be
# present and assemblable rather than vanishing into sampling noise.
FMT_BACKGROUND_MODELS = ["novaseq"]
FMT_BACKGROUND_N_READS = ["10M"]  # 10M total reads (R1+R2) = 5M read pairs

# Independent ISS runs per (model, sample, n_reads) combo, each with its own
# random seed -- same target abundance profile, but a different stochastic
# realization of it (per-genome read-count sampling around the target, plus
# per-read error simulation). Added 2026-09-22, per direction: without this,
# there was exactly one background per real FMT sample, so there was no way
# to separate genuine sample-to-sample (biological) variation from ISS's own
# simulation noise. Set back to 1 on 2026-10-24: only rep1 was ever generated
# (data/background_fmt/manifest.tsv seeds 1000-1002 are exactly what a
# 1-replicate run assigns), so this now matches what actually exists.
# Raise to 3 and rerun 07_generate_background_fmt.sh (--array=0-8) to add
# rep2/rep3 -- existing rep1 rows are skipped, but seeds are by array index,
# so a 3-replicate run would give the new reps different seeds than rep1's.
FMT_BACKGROUND_REPLICATES = 1

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


