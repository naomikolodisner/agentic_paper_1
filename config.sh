############################
# Base directories
############################

export XDISK=/xdisk/gwatts
export PROJECT_ROOT=$XDISK/kolodisner/agentic_paper_1
export RESULTS_ROOT=$PROJECT_ROOT/results
export DB_ROOT=/groups/gwatts/databases

############################
# Logs & scripts
############################

export WORK_DIR=$PROJECT_ROOT
export LOG_DIR=$PROJECT_ROOT/logs

############################
# Assembly inputs
############################

export XFILE=xac
export XFILE2=xad
export XFILE_DIR=$PROJECT_ROOT
export SPADES_DIR=$PROJECT_ROOT/out_spades

############################
# Viral detection tools
############################

export OUT_CHECKV=$RESULTS_ROOT/02_checkv
export CHECKVDB=$DB_ROOT/checkv-db-v1.5
export CHECKV_PARSER=$PROJECT_ROOT/CheckV_parser.R
export PARSE_LENGTH=5000

# VirSorter2
export OUT_VIRSORT=$RESULTS_ROOT/01_viral_detection/01A_virsorter2
export OUT_CHECKV_VIRSORT=$OUT_CHECKV/02A_virsorter2

# DeepVirFinder
export OUT_DVF=$RESULTS_ROOT/01_viral_detection/01B_dvf
export DVF_DB=$DB_ROOT/DeepVirFinder
export OUT_CHECKV_DVF=$OUT_CHECKV/02B_dvf

# geNomad
export OUT_GENOMAD=$RESULTS_ROOT/viral_detection/01C_genomad
export GENOMAD_DB=$DB_ROOT/genomad_db
export OUT_CHECKV_GENOMAD=$OUT_CHECKV/02C_genomad

# MARVEL
export OUT_MARVEL=$RESULTS_ROOT/viral_detection/01D_marvel
export MARVEL_DB=$DB_ROOT/MARVEL
export OUT_CHECKV_MARVEL=$OUT_CHECKV/02D_marvel

#export RSCRIPT_DIR=/groups/bhurwitz/miniconda3/bin/Rscript  '''redownload'''

# dereplication and clustering 
export OUT_DEREP=$RESULTS_ROOT/03_dereplicate
export OUT_CLUSTER=$RESULTS_ROOT/04_cluster

# step 1 create blastdb
export DB_DIR=$DB_ROOT/AVrC    
export MAX_DB_SIZE="0.5GB" 

# step 2 : blast query against blast db
export FASTA_DIR=$PROJECT_ROOT/query
export FA_SPLIT_FILE_SIZE=5000000 # in bytes, 5000 in KB

# BLAST parameters
export BLAST_TYPE=blastn
export MAX_TARGET_SEQS=1
export EVAL=1e-3
export OUT_FMT=6 # tabular format with no headings

# Annotation parameters
export PCTID=85
export LENGTH=1000
export BLAST_HITS="$RESULTS_ROOT/05D_mergeblast/AVrC_allrepresentatives.fasta/clusterRes_rep_seq.fasta.txt"
export ANNOTATIONS="$DB_ROOT/AVrC/database_csv/"
export OUTPUT="$RESULTS_ROOT/06_annotate"

#
# Some custom functions for our scripts
#
# --------------------------------------------------
function init_dir {
    for dir in $*; do
        if [ -d "$dir" ]; then
            rm -rf $dir/*
        else
            mkdir -p "$dir"
        fi
    done
}

# --------------------------------------------------
function create_dir {
    for dir in $*; do
        if [[ ! -d "$dir" ]]; then
          echo "$dir does not exist. Directory created"
          mkdir -p $dir
        fi
    done
}

# --------------------------------------------------
function lc() {
    wc -l $1 | cut -d ' ' -f 1
}
