# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **agentic benchmarking system** for viral detection in metagenomics. It evaluates up to 11 viral detection tools using an adaptive, multi-round pipeline that selects tools based on F1 score computed against a ground-truth dataset.

## Lab Notebook

`NOTEBOOK.md` at the project root is a running lab notebook. **Proactively append an entry**
(following its existing format) whenever you finish a meaningful chunk of work in a
session — a debugging pass, a pipeline/config change, an experiment run with results,
or before creating a commit. Skip trivial edits (typos, formatting). Don't ask first; just do
it, then mention it in your summary. Use the `notebook` skill if invoked explicitly via
`/notebook`.

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

**Dataset preparation (run before the main pipeline) — InSilicoSeq (ISS)-based, run from project root:**
```bash
python scripts/pick_refs.py               # Pick a random viral-genome subset (5/10/20 refs) per sample
sbatch scripts/generate_background.sh     # Build the background pool from HumGut/PHORAGER MAGs (48-combo array)
sbatch scripts/kraken2_sanity_check.sh    # Gate: confirm spiked viral refs are kraken2-detectable
sbatch scripts/read_mixing.sh             # Build every dist x percentage spike-in combo per sample
# then build the assembly work list and submit scripts/assemble.sh (array)

# or submit the whole chain with correct dependencies in one go:
bash scripts/submit_pipeline.sh
```
ART Illumina has been retired in favor of InSilicoSeq (ISS) — see "Synthetic Sample Creation
(ISS)" below. `scripts/remove_viral_contigs.py` and `data/no_virus_contigs/` /
`data/background_assemblies/` (the old ERR-marine-metagenome background) are superseded and no
longer used by the current scripts, but are left in place; `scripts/archive/` holds the retired
ART-era sample-creation scripts.

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

### Synthetic Sample Creation (ISS)

The spike-in benchmark dataset (`data/spike_in_samples/`) is built entirely with
[InSilicoSeq](https://insilicoseq.readthedocs.io/) (`insilicoseq` conda env), not ART — ART has
no realistic per-instrument error model and is Illumina/PacBio-agnostic in a way that doesn't
match real sequencers; ISS ships pre-built HiSeq/MiSeq/NextSeq/NovaSeq models (no PacBio).

- **Background**: `scripts/generate_background.sh` simulates the virus-free background
  metagenome from PHORAGER's prophage-stripped HumGut MAGs (`config.HUMGUT_PROPHAGE_REMOVED_DIR`
  — **not yet populated**; PHORAGER hasn't been run on HumGut yet, so this step fails fast with a
  clear message until that data lands). Each MAG file is passed as its own `--draft` argument —
  ISS treats each `--draft` *file* as one genome, so a single file concatenating many MAGs would
  wrongly be modeled as one organism (verified empirically 2026-07-07). Builds a 48-combo grid
  (4 `ISS_MODELS` x 4 `ISS_ABUNDANCE_DISTS` x 3 `BACKGROUND_N_READS`), logging each combo's
  **actual measured** read length and read-pair count to `config.BACKGROUND_MANIFEST` — read
  length is never hardcoded (the old ART code assumed 100bp; ISS models range 126–300bp).
- **Spike-in**: `scripts/pick_refs.py` draws a random subset of `VIRAL_SUBSET_SIZES` (5/10/20)
  AVrC viral genomes per sample dir (`sample_{size}v_{n}/`). `scripts/read_mixing.sh` then
  builds every `ISS_ABUNDANCE_DISTS` x `SPIKE_PERCENTAGES` (0.1%/1%/5%/10%) combo per sample: it
  picks a random background row, computes the exact viral `--n_reads` target for that percentage
  (`scripts/viral_spike_helper.py total-reads`), and calls
  `iss generate --genomes contigs.fasta --model {background's model} --abundance {dist}
  --n_reads {n_reads}` — ISS's own abundance distribution splits the subset's read budget across
  its genomes realistically (not evenly). Per-genome coverage is a **derived, logged** quantity
  computed after the fact from ISS's own `*_abundance.txt` output
  (`scripts/viral_spike_helper.py coverage-log`) — never an independent input.
  `scripts/combine_reads.py` concatenates + shuffles the viral spike with the background into
  `sample_{size}v_{n}/{dist}/spike_{pct}pct/final.{1,2}.fq.gz`.
- **Gate**: `scripts/kraken2_sanity_check.sh` runs once before read-mixing, confirming every
  viral reference used across the generated samples is actually detectable by kraken2. AVrC
  references use custom viral-catalog IDs (e.g. `GutCatV1_GPD_113896`), not NCBI accessions, so
  this can't be an ID-presence lookup against `seqid2taxid.map` — it simulates a small read set
  per reference and runs one combined kraken2 classify call, exiting nonzero if any reference has
  zero classified reads. In practice most AVrC genomes are novel/uncultured and get little or no
  hit against the general-purpose `kraken2_pluspfp` DB — that's an expected, meaningful finding,
  not a script bug.
- `scripts/assemble.sh` (MegaHit) is unchanged except for the FASTQ filename pattern.

### Configuration (`config.py`)

All paths, database locations, tool parameters, and output directory structure are centralized
here. `config.sh` (a shell mirror) exists but is stale and unused — no script sources it; treat
`config.py` as the single source of truth. Key variables:
- Project root: `/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1`
- Database root: `/rs1/researchers/b/blhurwit/users/nkolodi/databases`
- Tool root: `/rs1/researchers/b/blhurwit/users/nkolodi/tools`
- Input sample list: `XFILE_DIR / XFILE` → `data/sample_lists/xad` (sample IDs in `{profile}_{model}` format, e.g. `SRR4831655_hiseq`)
- Set3 simulated assemblies: `SPADES_DIR` → `data/set3_simulated_metagenomes/assemblies/`
- Background assemblies (ERR* marine metagenomes, superseded): `BACKGROUND_ASSEMBLIES` → `data/background_assemblies/`
- Ground truth: `CONTIG_IDENTITIES` → `data/set3_simulated_metagenomes/contig_identities.csv`
- ISS background/spike-in settings: `ISS_BIN`, `HUMGUT_PROPHAGE_REMOVED_DIR`, `BACKGROUND_ISS_DIR`,
  `BACKGROUND_MANIFEST`, `ISS_MODELS`, `ISS_ABUNDANCE_DISTS`, `BACKGROUND_N_READS`,
  `VIRAL_SUBSET_SIZES`, `SPIKE_PERCENTAGES`, `NUM_REPLICATES_PER_SUBSET_SIZE`
- Kraken2 sanity check: `KRAKEN2_DB`, `KRAKEN2_SEQID2TAXID`

Helper functions: `init_dir()`, `create_dir()`, `lc()` (line count).

### Data (`data/`)

| Directory | Contents |
|---|---|
| `set3_simulated_metagenomes/assemblies/` | Per-sample `assembly.fa` files; `{profile}_{model}` subdirs (e.g. `SRR4831655_hiseq`) |
| `set3_simulated_metagenomes/contig_identities.csv` | Ground-truth viral identity; columns: `profile`, `model`, `query_id`, `superkingdom` |
| `background_assemblies/` | Original ERR\* marine metagenome assemblies used as background in the old ART pipeline (superseded) |
| `no_virus_contigs/` | Background assemblies with geNomad-identified viral contigs stripped out (superseded by ISS + HumGut/PHORAGER) |
| `background_iss/` | ISS-simulated background reads, one dir per model/abundance/n_reads combo; `manifest.tsv` logs each combo's actual read length + pair count |
| `spike_in_samples/` | ISS-based spike-in samples: `sample_{size}v_{n}/{dist}/spike_{pct}pct/` |
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
| `test_env` | BioPython/pandas helper scripts (`pick_refs.py`, `viral_spike_helper.py`, `combine_reads.py`), MegaHit |
| `insilicoseq` | InSilicoSeq (ISS) — synthetic sample creation (background + viral spike-in reads) |

Each `@python_app` activates its specific environment via `conda run -n <env>` inside the function body.

## Key Files

| File | Purpose |
|---|---|
| `archive/agentic_viral_benchmark.py` | Original monolithic script — preserved, do not delete |
| `pipeline/coordinator.py` | Refactored entry point; orchestrates all agents |
| `pipeline/f1_eval.py` | F1/precision/recall; importable module and CLI |
| `pipeline/tool_selector.py` | Alpha-greedy tool selection |
| `config.py` | Python config — all paths, DB locations, parameters (source of truth; `config.sh` is stale/unused) |
| `run_viral_detection.sh` | SLURM submission wrapper — runs `pipeline/coordinator.py` |
| `scripts/pick_refs.py` | Picks a random viral-genome subset per spike-in sample dir |
| `scripts/generate_background.sh` | ISS background generation (model x abundance x depth grid) from HumGut/PHORAGER MAGs |
| `scripts/viral_spike_helper.py` | Computes viral `--n_reads` target from a percentage, and derives per-genome coverage post-hoc |
| `scripts/read_mixing.sh` | Builds every dist/percentage spike-in combo per sample via ISS |
| `scripts/combine_reads.py` | Concatenates + shuffles viral spike-in reads with background reads |
| `scripts/assemble.sh` | MegaHit assembly of each spike-in FASTQ pair |
| `scripts/kraken2_sanity_check.sh` | Gate: confirms spiked viral refs are kraken2-detectable before the heavier pipeline runs |
| `scripts/submit_pipeline.sh` | Submits the full ISS sample-creation chain with correct SLURM dependencies |
| `scripts/remove_viral_contigs.py` | Superseded — built the old ART-era ERR-marine-metagenome background |
| `scripts/archive/` | Retired ART-based sample-creation scripts |
| `scripts/CheckV_parser.R` | R script for parsing CheckV output |
| `scripts/experimental/` | In-progress scripts not yet part of the main pipeline |

## Logs

All logs are written to `logs/` with three subdirectories:
- `logs/slurm/` — SLURM job stdout/stderr (from `run_viral_detection.sh` and the ISS sample-creation scripts)
- `logs/sample/` — per-sample processing logs from the old ART-era `scripts/archive/create_single_equal_samples.sh`
- `logs/runinfo/` — Parsl workflow metadata (written automatically by the 4 Parsl configs)
