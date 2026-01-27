# Agentic Benchmarking
Agentic Benchmarking - Naomi, Mery, & Bonnie Spring 2026

## File Overview

- **`agentic_viral_benchmark.py`**  
  Main Python script that coordinates the viral detection workflow. Handles sample processing, tool selection, and benchmarking logic.

- **`CheckV_parser.R`**  
  R script to parse output from CheckV.

- **`config_py.sh`**  
  Shell script to set up variables before running the pipeline.

- **`run_viral_detection.sh`**  
  Shell script to launch the pipeline.

# Downloading All Databases and Environments Needed

## Genomad
- Github: https://github.com/apcamargo/genomad/blob/main/README.md

### Conda Install
    conda create -n genomad_env -c conda-forge -c bioconda genomad

### Testing
    conda activate genomad_env
    genomad --help

### Database
    genomad download-database /path/to/databases/directory

---

## Virsorter2
- Github: https://github.com/jiarong/VirSorter2

### Conda Install
    conda create -n virsorter2_env -c conda-forge -c bioconda virsorter=2

### Testing
    conda activate virsorter2_env
    virsorter --help

### Database
    virsorter setup -d db -j 4

---

## DeepVirFinder
- Github: https://github.com/jessieren/DeepVirFinder

### Conda Install
    conda create --name dvf_env python=3.6 numpy theano=1.0.3 keras=2.2.4 scikit-learn Biopython h5py=2.10.0 mkl-service=2.4.0

### Testing
    cd DeepVirFinder
    conda activate dvf_env
    python dvf.py --help

### Database
    git clone https://github.com/jessieren/DeepVirFinder
    cd DeepVirFinder

---

## CheckV
- Bitbucket: https://bitbucket.org/berkeleylab/checkv/src/master/README.md#markdown-header-installation
- Database portal: https://portal.nersc.gov/CheckV/

### Database
    wget -P /path/to/databases/directory https://portal.nersc.gov/CheckV/checkv-db-v1.5.tar.gz
    tar -xzvf checkv-db-v1.5.tar.gz

### Conda Install
    conda create -n checkv_env -c conda-forge -c bioconda checkv=1.0.1 -y

### Testing
    conda activate checkv_env
    checkv --help

---

## AVrC Database
- Github: https://github.com/aponsero/Aggregated_Viral_Catalogue/blob/main/README.md
- Database: https://zenodo.org/records/11426065

### Download
    wget -O /xdisk/bhurwitz/databases/AVrC_allrepresentatives.fasta.gz "https://zenodo.org/records/11426065/files/AVrC_allrepresentatives.fasta.gz?download=1"
    gunzip AVrC_allrepresentatives.fasta.gz
- Make a database directory (titled `AVrC`) and put the fasta file inside that directory.
- Download and unzip the csv annotation files as well within the AVrC directory
    wget -O database_csv.tar.gz https://zenodo.org/records/11426065/files/database_csv.tar.gz?download=1
    tar -xvzf database_csv.tar.gz
---

## Mmseqs
- Github: https://github.com/soedinglab/MMseqs2

### Conda Install
    conda create -n mmseqs2_env -c bioconda mmseqs2=13.45111

---

## Seqtk

### Conda Install
    conda create -n seqtk_env -c bioconda seqtk

---

## FaSplit
- Anaconda: https://anaconda.org/bioconda/ucsc-fasplit

### Conda Install
    conda create -n fasplit_env -c bioconda ucsc-fasplit

### Testing
    conda activate fasplit_env
    faSplit

---

## Blast
- Bioconda: https://bioconda.github.io/recipes/blast/README.html

### Conda Install
    conda create --name blast_env -c bioconda blast

### Testing
    conda activate blast_env
    blastn -version
