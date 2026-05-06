# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **agentic benchmarking system** for viral detection in metagenomics. It evaluates up to 11 viral detection tools using an adaptive, multi-round pipeline that selects tools based on performance metrics from CheckV quality assessment and BLAST annotation against the AVrC viral reference database.

## Running the Pipeline

**Primary entry point — submit the main SLURM job:**
```bash
sbatch run_viral_detection.sh
```
This activates the `academy_py311` conda environment and runs `agentic_viral_benchmark.py`.

**Dataset preparation (run before the main pipeline):**
```bash
python scripts/pick_refs.py                  # Select random viral reference sequences
python scripts/remove_viral_contigs.py       # Create background samples (no viral contigs)
sbatch scripts/create_single_equal_samples.sh  # Generate synthetic spike-in samples (SLURM array)
sbatch scripts/create_unequal_samples.sh       # Generate unequal-mixture samples
```

## Architecture

### Core Agent System (`agentic_viral_benchmark.py`)

The main script (~1,231 lines) orchestrates 10 rounds of benchmarking using the **Academy** agent framework (`from academy import Manager, Agent, action, loop`) combined with **Parsl** for distributed HPC execution.

**Agents:**
- `ViralDetectionAgent` — runs the selected viral detection tool on each sample
- `CheckVAgent` — runs CheckV quality assessment; computes `quality_ratio`
- `DereplicationClusteringAgent` — deduplicates and clusters detected sequences using MMseqs2
- `BLASTAgent` — annotates representative sequences against AVrC; computes `match_ratio`
- `CoordinatorAgent` — owns the 10-round loop; calls the others; drives tool selection

**Tool Selection (`ToolSelector` class):**
Uses an **alpha-greedy strategy** (α=0.6): with probability α, exploit the best-scoring tool; otherwise explore randomly. Score = `(quality_ratio + match_ratio) / 2`. Supports 11 tools: VirSorter, VirSorter2, DeepVirFinder, geNomad, MARVEL, VirFinder, VIBRANT, viralVerify, ViraMiner, MetaPhinder, Seeker.

**Parsl Configs (four, all SLURM on `standard` partition, account `gwatts`):**
- `viral_config` — 16 cores/worker, 4 workers/node, 12h walltime (heavy tool runs)
- `checkv_config` — 16 cores/worker, 3 workers/node, 4h walltime
- `derep_cluster_config` — 1 core/worker, 94 workers/node, 1h walltime
- `blast_config` — 1 core/worker, 94 workers/node, 1h walltime

### Configuration (`config.py` / `config.sh`)

All paths, database locations, tool parameters, and output directory structure are centralized here. Key variables:
- Project root: `/xdisk/gwatts/kolodisner/agentic_paper_1`
- Database root: `/groups/gwatts/databases`
- Tool root: `/groups/gwatts/tools`
- Input sample list: `xad` (text file of sample IDs)
- Assembly data: `data/set3_simulated_metagenomes/assemblies/`

Helper functions: `init_dir()`, `create_dir()`, `lc()` (line count).

### Dataset (`data/set3_simulated_metagenomes/`)

1,149,408 contigs from simulated reads using 3 error models (HiSeq, MiSeq, NovaSeq) and 5 abundance profiles from real marine metagenomes. Includes:
- `assemblies/` — per-sample `assembly.fa` files with taxonomic headers
- `bins/` — MetaBAT2 binned sequences
- `contig_identities.csv` — ground-truth viral identity from BLAST
- `previous_classifications.csv` — results from 9 existing tools (baseline comparison)

### Conda Environments

Each viral detection tool runs in its own isolated conda environment:

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
| `mmseqs2` | MMseqs2 (dereplication/clustering) |
| `fasplit_env` | FaSplit |
| `blast_env` | BLAST |
| `seqtk_env` | seqtk |
| `test_env` | ART Illumina simulator |

Each `@python_app` in `agentic_viral_benchmark.py` activates its specific environment via a subprocess call before running the tool.

## Key Files

| File | Purpose |
|---|---|
| `agentic_viral_benchmark.py` | Main orchestration; all agents, tool wrappers, and pipeline logic |
| `config.py` | Python config — all paths, DB locations, parameters |
| `config.sh` | Shell equivalent of config.py |
| `run_viral_detection.sh` | SLURM submission wrapper |
| `scripts/gen_titration_sample.py` | Mixes viral spike-in reads with background metagenome using seqtk |
| `scripts/pick_refs.py` | Randomly selects AVrC viral references for sample creation |
| `scripts/remove_viral_contigs.py` | Creates background samples by stripping geNomad-identified viral contigs |
| `scripts/CheckV_parser.R` | R script for parsing CheckV output |
