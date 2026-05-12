# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **agentic benchmarking system** for viral detection in metagenomics. It evaluates up to 11 viral detection tools using an adaptive, multi-round pipeline that selects tools based on F1 score computed against a ground-truth dataset.

## Running the Pipeline

**Refactored pipeline (active development) — run from project root:**
```bash
python -m pipeline.coordinator
# or submit via SLURM:
sbatch run_viral_detection.sh
```

**Original monolithic script (preserved, do not delete):**
```bash
python archive/agentic_viral_benchmark.py
```

**Dataset preparation (run before the main pipeline):**
```bash
python scripts/pick_refs.py                    # Select random viral reference sequences
python scripts/remove_viral_contigs.py         # Create background samples (no viral contigs)
sbatch scripts/create_single_equal_samples.sh  # Generate synthetic spike-in samples (SLURM array)
sbatch scripts/create_unequal_samples.sh       # Generate unequal-mixture samples
```

**Run F1 evaluation standalone:**
```bash
python -m pipeline.f1_eval path/to/viral.fasta \
    --tool seeker --sample-id SRR4831655_hiseq \
    --profile SRR4831655 --model hiseq
```

## Architecture

### Refactored Pipeline (`pipeline/`)

The pipeline is split into focused modules. `coordinator.py` is the entry point and imports from all others.

```
pipeline/
├── coordinator.py       # CoordinatorAgent, process_sample(), main()
├── viral_detection.py   # @python_app wrappers for all 11 tools + ViralDetectionAgent
├── checkv.py            # checkv_app + CheckVAgent
├── derep_cluster.py     # MMseqs2 apps + DereplicationClusteringAgent
├── blast.py             # BLAST apps + BLASTAgent + annotate_blast()
├── f1_eval.py           # F1 / precision / recall evaluation
├── tool_selector.py     # ToolSelector (alpha-greedy)
└── parsl_configs.py     # The 4 Parsl SLURM configs
```

**Agents:**
- `ViralDetectionAgent` — runs the selected viral detection tool on each sample
- `CheckVAgent` — runs CheckV quality assessment; computes `quality_ratio` (logged, not used for scoring)
- `DereplicationClusteringAgent` — deduplicates and clusters detected sequences using MMseqs2
- `BLASTAgent` — annotates representative sequences against AVrC; computes `match_ratio` (logged, not used for scoring)
- `CoordinatorAgent` — owns the 10-round loop; drives tool selection via F1 score

**Tool Selection (`ToolSelector` class in `pipeline/tool_selector.py`):**
Uses an **alpha-greedy strategy** (α=0.6): with probability α exploit the best-scoring tool, otherwise explore randomly. **F1 score is the primary evaluation signal.** `quality_ratio` and `match_ratio` are still computed each round and logged with `(future use)` labels for potential future use. Supports 11 tools: VirSorter, VirSorter2, DeepVirFinder, geNomad, MARVEL, VirFinder, VIBRANT, viralVerify, ViraMiner, MetaPhinder, Seeker.

**F1 Evaluation (`pipeline/f1_eval.py`):**
Computes precision, recall, and F1 against `contig_identities.csv`. Ground truth is filtered to the specific `profile` and `model` for each sample — this is critical for accurate recall. The sample directory name encodes both: `SRR4831655_hiseq` → `profile="SRR4831655"`, `model="hiseq"`. The CSV (~1.1 M rows) is loaded once per unique `(profile, model)` pair per round and cached.

Thresholds: F1 ≥ 0.70 = good, F1 < 0.50 = terrible (drop configuration).

Three tools return non-FASTA output (VIBRANT returns a directory, viralVerify returns a directory, ViraMiner returns a predictions TXT). F1 evaluation is skipped for these until per-tool FASTA extraction is implemented — see `_resolve_viral_fasta()` in `coordinator.py`.

**Parsl Configs (`pipeline/parsl_configs.py`, all SLURM on `standard` partition, account `gwatts`):**
- `viral_config` — 16 cores/worker, 4 workers/node, 12h walltime (heavy tool runs)
- `checkv_config` — 16 cores/worker, 3 workers/node, 4h walltime
- `derep_cluster_config` — 1 core/worker, 94 workers/node, 1h walltime
- `blast_config` — 1 core/worker, 94 workers/node, 1h walltime

### Configuration (`config.py` / `config.sh`)

All paths, database locations, tool parameters, and output directory structure are centralized here. Key variables:
- Project root: `/xdisk/gwatts/kolodisner/agentic_paper_1`
- Database root: `/groups/gwatts/databases`
- Tool root: `/groups/gwatts/tools`
- Input sample list: `XFILE_DIR / XFILE` → `data/sample_lists/xad` (sample IDs in `{profile}_{model}` format, e.g. `SRR4831655_hiseq`)
- Set3 simulated assemblies: `SPADES_DIR` → `data/set3_simulated_metagenomes/assemblies/`
- Background assemblies (ERR* marine metagenomes): `BACKGROUND_ASSEMBLIES` → `data/background_assemblies/`
- Ground truth: `CONTIG_IDENTITIES` → `data/set3_simulated_metagenomes/contig_identities.csv`

Helper functions: `init_dir()`, `create_dir()`, `lc()` (line count).

### Data (`data/`)

| Directory | Contents |
|---|---|
| `set3_simulated_metagenomes/assemblies/` | Per-sample `assembly.fa` files; `{profile}_{model}` subdirs (e.g. `SRR4831655_hiseq`) |
| `set3_simulated_metagenomes/contig_identities.csv` | Ground-truth viral identity; columns: `profile`, `model`, `query_id`, `superkingdom` |
| `background_assemblies/` | Original ERR\* marine metagenome assemblies used as background in spike-in creation |
| `no_virus_contigs/` | Background assemblies with geNomad-identified viral contigs stripped out |
| `spike_in_samples/` | Synthetic spike-in samples at varying coverages and compositions |
| `sample_lists/xad` | Sample IDs for the active pipeline run (one `{profile}_{model}` per line) |
| `sample_lists/xac` | Alternate sample ID list |

Assemblies use SPAdes sequential naming (`k141_0`, `k141_1`, …) independently per community — the same contig ID can appear in every assembly but refers to a different sequence each time. Always filter `contig_identities.csv` by `profile` + `model` for accurate F1 recall.

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

Each `@python_app` activates its specific environment via `conda run -n <env>` inside the function body.

## Key Files

| File | Purpose |
|---|---|
| `archive/agentic_viral_benchmark.py` | Original monolithic script — preserved, do not delete |
| `pipeline/coordinator.py` | Refactored entry point; orchestrates all agents |
| `pipeline/f1_eval.py` | F1/precision/recall; importable module and CLI |
| `pipeline/tool_selector.py` | Alpha-greedy tool selection |
| `config.py` | Python config — all paths, DB locations, parameters |
| `config.sh` | Shell equivalent of config.py |
| `run_viral_detection.sh` | SLURM submission wrapper — runs `pipeline/coordinator.py` |
| `scripts/gen_titration_sample.py` | Mixes viral spike-in reads with background metagenome using seqtk |
| `scripts/pick_refs.py` | Randomly selects AVrC viral references for sample creation |
| `scripts/remove_viral_contigs.py` | Creates background assemblies by stripping geNomad-identified viral contigs |
| `scripts/CheckV_parser.R` | R script for parsing CheckV output |
| `scripts/experimental/` | In-progress scripts not yet part of the main pipeline |

## Logs

All logs are written to `logs/` with three subdirectories:
- `logs/slurm/` — SLURM job stdout/stderr (from `run_viral_detection.sh` and `scripts/create_*.sh`)
- `logs/sample/` — per-sample processing logs from `create_single_equal_samples.sh`
- `logs/runinfo/` — Parsl workflow metadata (written automatically by the 4 Parsl configs)
