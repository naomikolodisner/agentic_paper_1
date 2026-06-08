# Agentic Viral Benchmarking
Agentic Benchmarking - Naomi, Mery, & Bonnie Spring 2026

## Overview

An agentic pipeline that benchmarks up to 10 viral detection tools on synthetic spike-in metagenome assemblies. Each run randomly selects a sample type and coverage level, then runs a 10-round adaptive loop that scores tools by F1 and uses an alpha-greedy strategy to allocate future rounds toward better-performing tools.

## Running the Pipeline

```bash
# Run directly
python -m pipeline.coordinator

# Submit to SLURM
sbatch run_viral_detection.sh
```

The original monolithic script is preserved at `archive/agentic_viral_benchmark.py` — do not delete it.

## Architecture

### Pipeline Modules (`pipeline/`)

| Module | Role |
|---|---|
| `coordinator.py` | Entry point; owns the 10-round loop and drives all agents |
| `viral_detection.py` | Parsl `@python_app` wrappers for all detection tools + `ViralDetectionAgent` |
| `checkv.py` | CheckV quality assessment + `CheckVAgent` |
| `derep_cluster.py` | MMseqs2 dereplication and clustering + `DereplicationClusteringAgent` |
| `blast.py` | BLAST annotation against AVrC + `BLASTAgent` |
| `f1_eval.py` | F1 / precision / recall evaluation; also a standalone CLI |
| `tool_selector.py` | Alpha-greedy `ToolSelector` |
| `parsl_configs.py` | Four named Parsl SLURM executor configs |

### The Loop

**Initialization (once per run):**
- `select_spike_in_samples()` randomly picks a sample **type** (`single`, `equal2`, `equal3`, `equal4`, `unequal2`, `unequal3`) and a **coverage level** common to all 15 samples of that type (e.g. `10x`, `0.5x`, `1x_0.5x`).
- The selected type and coverage are logged and held fixed for all 10 rounds.

**Each round:**
1. **Tool selection** — alpha-greedy (α=0.6): exploit the best-scoring tool with probability 0.6, otherwise explore randomly from the 10-tool pool.
2. **Viral detection** — all 15 spike-in assemblies run in parallel on the selected tool via Parsl/SLURM.
3. **F1 evaluation** — predicted viral contigs vs. ground truth; average F1 across samples is the scoring signal.
4. **CheckV** — quality assessment per sample; `quality_ratio` is logged for future use.
5. **Dereplication** — MMseqs2 deduplication per sample.
6. **Tool score update** — average F1 fed back into the selector for the next round.
7. **Clustering + BLAST** (skipped if no dereplicated FASTA) — cross-sample clustering, then BLAST against AVrC; `match_ratio` is logged for future use.

After 10 rounds the pipeline reports the best tool and average F1 then shuts down.

### Tool Pool (10 tools)

VirSorter, VirSorter2, DeepVirFinder, geNomad, MARVEL, VirFinder, VIBRANT, viralVerify, MetaPhinder, Seeker

ViraMiner is excluded from scoring because its predictions TXT output has no FASTA extraction yet.

### Sample Input (`data/spike_in_samples/`)

Synthetic spike-in metagenome assemblies organized as:
```
data/spike_in_samples/
├── single/          # 1 viral reference spiked in
├── equal2/          # 2 viral refs at equal coverage
├── equal3/
├── equal4/
├── unequal2/        # 2 viral refs at unequal coverage (e.g. 10x_1x)
└── unequal3/
    └── sample{1-15}/
        ├── refs_log.txt        # viral references used
        ├── contigs.fasta       # viral reference sequences
        └── {coverage}/
            ├── assembly.fa     # SPAdes assembled metagenome
            └── sample.{1,2}.fq.gz
```

F1 evaluation is currently 0 for spike-in samples (no `contig_identities.csv` ground truth available); tool selection runs in explore mode throughout.

### Configuration (`config.py`)

All paths and parameters are centralized. Key variables:

| Variable | Path |
|---|---|
| `SPIKE_IN_DIR` | `data/spike_in_samples/` |
| `SPADES_DIR` | `data/set3_simulated_metagenomes/assemblies/` (legacy) |
| `CONTIG_IDENTITIES` | `data/set3_simulated_metagenomes/contig_identities.csv` |
| `DB_ROOT` | `users/nkolodi/databases/` |
| `RESULTS_ROOT` | `results/` |
| `LOG_DIR` | `logs/` |

### Conda Environments

Each tool runs in its own isolated environment via `conda run -n <env>`:

| Environment | Tool(s) |
|---|---|
| `academy_py311` | Main orchestrator (Academy, Parsl, BioPython, pandas) |
| `virsorter2_env` | VirSorter2 |
| `dvf_env` | DeepVirFinder (Python 3.6) |
| `genomad_env` | geNomad |
| `marvel_env` | MARVEL |
| `virfinder` | VirFinder (R-based) |
| `vibrant` | VIBRANT |
| `checkv_env` | CheckV |
| `mmseqs2` | MMseqs2 |
| `blast_env` | BLAST |
| `seqtk_env` | seqtk |
| `fasplit_env` | FaSplit |

### Logs

| Directory | Contents |
|---|---|
| `logs/slurm/` | SLURM job stdout/stderr |
| `logs/sample/` | Per-sample processing logs |
| `logs/runinfo/` | Parsl workflow metadata |

---

# Installing Databases and Environments

## geNomad
- GitHub: https://github.com/apcamargo/genomad

```bash
conda create -n genomad_env -c conda-forge -c bioconda genomad
conda activate genomad_env
genomad download-database /path/to/databases/
```

---

## VirSorter2
- GitHub: https://github.com/jiarong/VirSorter2

```bash
conda create -n virsorter2_env -c conda-forge -c bioconda virsorter=2
conda activate virsorter2_env
virsorter setup -d db -j 4
```

---

## DeepVirFinder
- GitHub: https://github.com/jessieren/DeepVirFinder

```bash
conda create --name dvf_env python=3.6 numpy theano=1.0.3 keras=2.2.4 scikit-learn Biopython h5py=2.10.0 mkl-service=2.4.0
git clone https://github.com/jessieren/DeepVirFinder
```

---

## CheckV
- Bitbucket: https://bitbucket.org/berkeleylab/checkv
- Database portal: https://portal.nersc.gov/CheckV/

```bash
wget -P /path/to/databases/ https://portal.nersc.gov/CheckV/checkv-db-v1.5.tar.gz
tar -xzvf checkv-db-v1.5.tar.gz
conda create -n checkv_env -c conda-forge -c bioconda checkv=1.0.1 -y
```

---

## AVrC Database
- GitHub: https://github.com/aponsero/Aggregated_Viral_Catalogue
- Download: https://zenodo.org/records/11426065

```bash
wget -O /path/to/databases/AVrC/AVrC_allrepresentatives.fasta.gz \
    "https://zenodo.org/records/11426065/files/AVrC_allrepresentatives.fasta.gz?download=1"
gunzip AVrC_allrepresentatives.fasta.gz
wget -O database_csv.tar.gz https://zenodo.org/records/11426065/files/database_csv.tar.gz?download=1
tar -xvzf database_csv.tar.gz
```

---

## MMseqs2
- GitHub: https://github.com/soedinglab/MMseqs2

```bash
conda create -n mmseqs2_env -c bioconda mmseqs2=13.45111
```

---

## seqtk

```bash
conda create -n seqtk_env -c bioconda seqtk
```

---

## FaSplit
- Anaconda: https://anaconda.org/bioconda/ucsc-fasplit

```bash
conda create -n fasplit_env -c bioconda ucsc-fasplit
```

---

## BLAST
- Bioconda: https://bioconda.github.io/recipes/blast/README.html

```bash
conda create --name blast_env -c bioconda blast
```
