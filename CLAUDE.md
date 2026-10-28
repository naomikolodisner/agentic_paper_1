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

## Project Layout

The repo is two pipelines with the same layout, sharing the root `config.py` (single source of truth):

```
agentic_paper_1/
├── config.py                 # shared config; both pipelines import it from the root
├── mock_datasets/            # Pipeline 1: mock dataset creation (complete). See mock_datasets/README.md
│   ├── README.md             #   step order + commands
│   ├── scripts/              #   steps numbered in run order, 01_ ... 12_
│   │   └── helpers/          #   called by a step, never run directly
│   ├── archive/              #   retired scripts
│   └── results/              #   intermediate outputs (config.OUT_KRAKEN2_FMT_READS etc.)
├── agentic_benchmarking/     # Pipeline 2: agentic benchmarking (in progress)
│   ├── README.md
│   ├── pipeline/             #   coordinator + agents (Python package, run as `python -m pipeline.coordinator`)
│   ├── scripts/              #   run_viral_detection.sh (SLURM wrapper), CheckV_parser.R
│   ├── archive/              #   agentic_viral_benchmark.py (original monolith, do not delete)
│   └── results/              #   numbered stage outputs (config.RESULTS_ROOT)
├── data/                     # shared data: mock_datasets inputs + the spike-in samples (the benchmark's input)
└── logs/                     # shared logs (slurm/, runinfo/, sample/)
```

Both `results/` dirs are gitignored. Python code finds `config.py` with
`sys.path.insert(0, <project root>)` (`.parent` chains: 3 levels for `scripts/*.py` and
`pipeline/*.py`, 4 for `scripts/helpers/*.py`). A script moved to a different depth must change
its chain. Only the scripts the current pipeline actually runs live in `scripts/`. Anything
retired goes to that pipeline's `archive/`.

## Running the Pipeline

**Agentic benchmarking (active development) — run from `agentic_benchmarking/`:**
```bash
cd agentic_benchmarking
python -m pipeline.coordinator
# or submit via SLURM (from the project root):
sbatch agentic_benchmarking/scripts/run_viral_detection.sh
```

**Original monolithic script (preserved, do not delete):**
```bash
python agentic_benchmarking/archive/agentic_viral_benchmark.py
```

**Mock dataset creation (run before the benchmark), from the project root.** Steps are numbered in
run order. The full table (submit commands, array sizes, outputs) is in `mock_datasets/README.md`:
```
01_kraken2_fmt_reads.sh -> 02_bracken_build.sh (one-time) -> 03_bracken_run.sh
  -> 04_pick_representative_genomes.py -> 05_extract_humgut_phoraged.sh
  -> 06_build_fmt_abundance_profile.py -> 07_generate_background_fmt.sh     # background, done for 3 samples
08_check_kraken2_accessions.py -> 09_pick_refs.py --design og -> 10_read_mixing.sh
  -> 11_kraken2_check.sh (gate) -> 12_assemble.sh                           # spike-in samples
```
There is no one-shot submit script any more: `submit_pipeline.sh` was archived (now in `mock_datasets/archive/`)
on 2026-10-24 (it chained the retired background step and percentage design); job chaining will
come from the Nextflow port.
Viral spike-in read simulation uses **ART Illumina** (reintroduced deliberately, scoped to
spike-in only), mixed with a background pool. `mock_datasets/archive/` holds retired sample-creation
scripts, including the original ISS-built-in-distribution `generate_background.sh` +
`extract_humgut_subset.py` (superseded by the FMT-profile approach) and the older
equal2/equal3/equal4/unequal2/unequal3-era scripts that `10_read_mixing.sh`'s ART coverage logic is
modeled on. `data/archive/` holds their old output.

**Run F1 evaluation standalone (from `agentic_benchmarking/`):**
```bash
python -m pipeline.f1_eval path/to/viral.fasta \
    --tool seeker --sample-id SRR4831655_hiseq \
    --profile SRR4831655 --model hiseq
```

## Architecture

### Refactored Pipeline (`agentic_benchmarking/pipeline/`)

The pipeline is split into focused modules. `coordinator.py` is the entry point and imports from all others.

```
agentic_benchmarking/pipeline/
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

**Tool Selection (`ToolSelector` class in `agentic_benchmarking/pipeline/tool_selector.py`):**
Uses an **alpha-greedy strategy** (α=0.6): with probability α exploit the best-scoring tool, otherwise explore randomly. **F1 score is the primary evaluation signal.** `quality_ratio` and `match_ratio` are still computed each round and logged with `(future use)` labels for potential future use. Supports 11 tools: VirSorter, VirSorter2, DeepVirFinder, geNomad, MARVEL, VirFinder, VIBRANT, viralVerify, ViraMiner, MetaPhinder, Seeker.

**F1 Evaluation (`agentic_benchmarking/pipeline/f1_eval.py`):**
Computes precision, recall, and F1 against `contig_identities.csv`. Ground truth is filtered to the specific `profile` and `model` for each sample — this is critical for accurate recall. The sample directory name encodes both: `SRR4831655_hiseq` → `profile="SRR4831655"`, `model="hiseq"`. The CSV (~1.1 M rows) is loaded once per unique `(profile, model)` pair per round and cached.

Thresholds: F1 ≥ 0.70 = good, F1 < 0.50 = terrible (drop configuration).

Three tools return non-FASTA output (VIBRANT returns a directory, viralVerify returns a directory, ViraMiner returns a predictions TXT). F1 evaluation is skipped for these until per-tool FASTA extraction is implemented — see `_resolve_viral_fasta()` in `coordinator.py`.

**Parsl Configs (`agentic_benchmarking/pipeline/parsl_configs.py`, all SLURM on `standard` partition, account `gwatts`):**
- `viral_config` — 16 cores/worker, 4 workers/node, 12h walltime (heavy tool runs)
- `checkv_config` — 16 cores/worker, 3 workers/node, 4h walltime
- `derep_cluster_config` — 1 core/worker, 94 workers/node, 1h walltime
- `blast_config` — 1 core/worker, 94 workers/node, 1h walltime

### Synthetic Sample Creation

The background pool is built with [InSilicoSeq](https://insilicoseq.readthedocs.io/)
(`insilicoseq` conda env) for its realistic per-instrument error models (ISS ships pre-built
HiSeq/MiSeq/NextSeq/NovaSeq models; ART has no such per-instrument model and is
Illumina/PacBio-agnostic in a way that doesn't match real sequencers). Viral spike-in reads are
simulated with **ART Illumina** instead (`test_env` conda env) — reintroduced deliberately,
scoped to spike-in only, mirroring the retired `mock_datasets/archive/`
equal2/equal3/equal4/unequal2/unequal3 scripts' fixed-coverage-per-genome approach — then mixed
with a randomly chosen background row from the ISS pool.

- **Background (being replaced — in progress)**: the original approach,
  `mock_datasets/archive/generate_background.sh` (+ `mock_datasets/archive/extract_humgut_subset.py`),
  simulated the virus-free background from a random subset of raw HumGut MAGs using ISS's
  built-in `lognormal`/`exponential` abundance distributions. **Archived** — output preserved at
  `data/archive/background_iss/`. It's being replaced with a background driven by a **real**
  microbiome abundance profile instead: classify an actual FMT sample assembly against a custom
  HumGut-only Kraken2 DB (`config.HUMGUT_KRAKEN2_DB` → `data/humgut_kraken2/`, GTDB taxonomy, no
  human library, built 2026-09-08), turn the classification into a per-genome abundance profile
  (`Σ(contig_length × SPAdes_coverage)` per HumGut genome, normalized), and simulate the
  background from that profile using PHORAGER's prophage-stripped ("phoraged") HumGut genomes
  instead of raw ones. **Current status (as of 2026-09-22, needs a fuller pass)**:
  `mock_datasets/archive/kraken2_fmt.sh` (contig-based classification, ERR2198611–613) is **archived** —
  superseded 2026-09-19 by read-based classification (`mock_datasets/scripts/01_kraken2_fmt_reads.sh` against
  `data/sample_lists/xae`: ERR2198703–705, whose real decontaminated reads are on Hazel, unlike
  611–613 which only ever had a SPAdes assembly here). The full chain from there — Bracken
  correction (`mock_datasets/scripts/02_bracken_build.sh` → `mock_datasets/scripts/03_bracken_run.sh`, fixing kraken2's ~32%
  ancestor-rank-collapsed reads), `mock_datasets/scripts/04_pick_representative_genomes.py`,
  `mock_datasets/scripts/06_build_fmt_abundance_profile.py`, `mock_datasets/scripts/05_extract_humgut_phoraged.sh`, and
  `mock_datasets/scripts/07_generate_background_fmt.sh` — is built and **has already run successfully**:
  `data/background_fmt/manifest.tsv` has real novaseq/10M-read output for all 3 samples. `10_read_mixing.sh` was
  repointed at `data/background_fmt/` (`config.OUT_BACKGROUND_FMT`) 2026-10-06. One background
  replicate per sample (`FMT_BACKGROUND_REPLICATES = 1`; rep2/rep3 were never generated).
  Any new background-generation script must still respect the same ISS constraint the old one
  did: each `--draft` *file* is treated as one genome, so a single file concatenating many
  genomes together would wrongly be modeled as one organism (verified empirically 2026-07-07).
- **Spike-in** (`--design og`, current): an exact replica of the PI's original ART scripts
  (`mock_datasets/archive/` single/equal/unequal), with genomes drawn fresh from **INPHARED** and the
  FMT background. `config.SPIKE_OG_DESIGNS` defines six types — `single`, `equal2`, `equal3`,
  `equal4` (each at 0.1x/0.5x/1x/10x, every genome at the same coverage), `unequal2`
  (10x_1x, 1x_0.5x) and `unequal3` (10x_1x_0.5x, 1x_0.5x_0.1x). `mock_datasets/scripts/09_pick_refs.py --design og`
  writes `{type}/sample{n}/` with `contigs.fasta`, `refs_log.txt` and `conditions.tsv` (condition
  name, then per-genome ART coverages in `contigs.fasta` order). `SPIKE_OG_REPLICATES` is **1**
  for initial testing (6 sample dirs, 20 conditions); the original used 15. Genomes come from the
  408 accessions (`config.INPHARED_ACCESSIONS_LIST` → `data/accessions_in_kraken2.txt`) of the
  ~2902-genome INPHARED catalog whose IDs appear in the Kraken2 DB's `seqid2taxid.map`
  (`mock_datasets/scripts/08_check_kraken2_accessions.py` — an ID lookup, no classification). Each accession is its
  own FASTA in `config.INPHARED_GENOMES_DIR` (`/rs1/shares/brc/databases/inphared/GenomesDB`).
  AVrC (`config.AVRC_ALL_SEQUENCES` / `AVRC_METADATA_CSV`) is no longer the spike-in source; it
  still backs the BLAST annotation step (`config.DB_DIR` / `ANNOTATIONS`).
  `mock_datasets/scripts/10_read_mixing.sh` (one array task per `conditions.tsv`) then, per condition: draws one
  random usable background row (`VALID_BG_ROWS`, filtered by read length vs.
  `ART_PROFILE_BY_MODEL` / `ART_PROFILE_MAX_LEN`); runs `art_illumina -ss MSv3 -p -na -f {cov}
  -l {background read length, 151} -m 200 -s 10` per genome (MSv3 because ART has no NovaSeq
  profile); logs actual per-genome coverage from ART's read counts
  (`mock_datasets/scripts/helpers/viral_spike_helper.py log-actual-coverage` → `coverage_log.txt`, QC only) plus
  `spike_summary.tsv`; and `mock_datasets/scripts/helpers/combine_reads.py` shuffles viral + background into
  `{type}/sample{n}/{condition}/final.{1,2}.fq.gz`. The viral read % is derived, never an input.
  The older large-subset design (`--design subsets`, `VIRAL_SUBSET_SIZES` [80, 115, 150],
  `SPIKE_COVERAGES`) is still in `09_pick_refs.py` but unused. The retired percentage design's
  code (`viral_spike_helper.py`'s `total-reads` / `weighted-coverage-plan`, `SPIKE_PERCENTAGES`,
  `ART_RATIO_WEIGHTS`) was removed 2026-10-27.
- **Gate**: `mock_datasets/scripts/11_kraken2_check.sh` runs after read-mixing but before assembly, confirming
  every viral reference actually spiked into the generated samples is detectable by kraken2
  using the **real** post-mix reads (not a synthetic simulation). It pools every sample/pct
  combo's `final.1.fq.gz` as-is (background reads included) and runs ONE combined kraken2
  classify call over everything, then attributes each classified/unclassified read back to its
  source genome by parsing kraken2's own read-ID column — reads whose derived genome ID isn't
  one of the sample's known viral references (i.e. background reads) are ignored. This relies
  on ART's own read-naming convention for the viral reads (`<genome_id>-<read_number>/1`,
  verified empirically 2026-07-15 via a live `art_illumina` test), which `combine_reads.py`'s
  shuffle leaves untouched and kraken2's `--output` echoes verbatim — background reads keep
  ISS's own convention, but that's irrelevant since background genome IDs never match a known
  viral reference anyway. It's deliberately agnostic to
  whichever viral reference catalog built the spike-in samples (AVrC, INPHARED, or otherwise) —
  no hardcoded catalog FASTA or ID scheme, just whatever `refs_log.txt` `09_pick_refs.py` already
  wrote. Some catalogs use custom IDs (e.g. AVrC's `GutCatV1_GPD_113896`) that don't overlap
  NCBI accessions at all, so this can't be a general ID-presence lookup against
  `seqid2taxid.map` either. Exits nonzero if any reference has zero classified reads (reported
  separately from references that got zero real spike-in reads at all, which is a
  data-generation issue, not a detectability one). The current INPHARED-based catalog was
  specifically chosen as the 408 accessions already confirmed present in `seqid2taxid.map`, so
  high classification rates are the expected outcome here — this is a different situation from
  AVrC (superseded as the spike-in source), whose largely novel/uncultured genomes got little or
  no hit against the general-purpose `kraken2_pluspfp` DB; a low-recall result for the current
  catalog would be a real signal worth investigating, not an expected finding.
- `mock_datasets/scripts/12_assemble.sh` (MegaHit) is unchanged except for the FASTQ filename pattern.

### Configuration (`config.py`)

All paths, database locations, tool parameters, and output directory structure are centralized
here; it is the single source of truth (shell scripts read values from it via `python -c`). The
old `config.sh` shell mirror, which still pointed at the pre-Hazel `/xdisk` paths, was deleted 2026-10-27. Key variables:
- Project root: `/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1`
- Database root: `/rs1/researchers/b/blhurwit/users/nkolodi/databases`
- Tool root: `/rs1/researchers/b/blhurwit/users/nkolodi/tools`
- Input sample list: `XFILE_DIR / XFILE` → `data/sample_lists/xad` (sample IDs in `{profile}_{model}` format, e.g. `SRR4831655_hiseq`)
- **Archived** (moved to a clearly labeled section in `config.py`, kept so the archived scripts
  in `mock_datasets/archive/` still import cleanly): `SPADES_DIR` / `CONTIG_IDENTITIES` (old Set3
  ground truth, repointed at `data/archive/set3_simulated_metagenomes/` — no longer used at all,
  the new FMT-derived datasets are the ground truth going forward), `HUMGUT_PROPHAGE_REMOVED_DIR`,
  `HUMGUT_TAR`, `HUMGUT_TSV`, `HUMGUT_SUBSET_N`, `HUMGUT_SUBSET_SEED`, `HUMGUT_RAW_MAGS_DIR`,
  `BACKGROUND_ISS_DIR` / `BACKGROUND_MANIFEST` (repointed at `data/archive/background_iss/`),
  `ISS_MODELS`, `BACKGROUND_ABUNDANCE_DISTS`, `BACKGROUND_N_READS`
- Background assemblies (ERR* marine metagenomes, superseded/archived): `BACKGROUND_ASSEMBLIES` → `data/archive/background_assemblies/`
- FMT-derived background (active, in progress — see "Synthetic Sample Creation"):
  `ISS_BIN`, `HUMGUT_KRAKEN2_DB` → `data/humgut_kraken2/`, `FMT_ASSEMBLIES_DIR`,
  `FMT_SAMPLE_LIST` → `data/sample_lists/xac`, `OUT_KRAKEN2_FMT` → `mock_datasets/results/kraken2_fmt/`
- Spike-in settings (ART-based): `SPIKE_OG_DESIGNS`, `SPIKE_OG_REPLICATES` (active `og` design);
  `VIRAL_SUBSET_SIZES`, `SPIKE_COVERAGES` (unused subsets design),
  `NUM_REPLICATES_PER_SUBSET_SIZE`, `ART_BIN`, `ART_FRAGMENT_MEAN`, `ART_FRAGMENT_SD`,
  `ART_PROFILE_BY_MODEL`, `ART_PROFILE_MAX_LEN`
- Spike-in genome catalog (INPHARED-based, full ~2902-genome catalog): `INPHARED_ACCESSIONS_LIST`
  → `data/accessions_in_kraken2.txt` (408 confirmed), `INPHARED_GENOMES_DIR` →
  `/rs1/shares/brc/databases/inphared/GenomesDB` (`AVRC_ALL_SEQUENCES` / `AVRC_METADATA_CSV` are
  unused by spike-in now, kept for `DB_DIR`/`ANNOTATIONS`'s unrelated BLAST-annotation use)
- Kraken2 check gate (spike-in detectability, general-purpose DB): `KRAKEN2_DB`, `KRAKEN2_SEQID2TAXID`
  (`/rs1/shares/brc/admin/databases/kraken2_pluspfp`). Despite the folder name, this is
  **Kraken2 PlusPF, 16 GB capped, release 2025-10-15** (`k2_pluspf_16_GB_20251015.tar.gz`;
  `library_report.tsv` has no plant library) — use that name in methods.

Helper functions: `init_dir()`, `create_dir()`, `lc()` (line count).

### Data (`data/`)

**`data/archive/`** holds everything superseded and no longer read by the active pipeline —
including `set3_simulated_metagenomes/` (the old ground truth; **no longer used at all**, the
new FMT-derived datasets are the ground truth going forward), `background_iss/` (the old
built-in-ISS-distribution background), `background_assemblies/`, `no_virus_contigs/`, and the
even-older `equal2/3/4`/`unequal2/3`/`single` ART-only pipeline output. See
`data/archive/README.md`.

| Directory | Contents |
|---|---|
| `humgut_kraken2/` | Custom HumGut-only Kraken2 DB (GTDB taxonomy, no human library), built 2026-09-08 + Bracken index (`mock_datasets/scripts/02_bracken_build.sh`). Used by `mock_datasets/scripts/01_kraken2_fmt_reads.sh` to classify real FMT reads for the FMT abundance profile — separate from `KRAKEN2_DB`/`kraken2_pluspfp`, which only backs the spike-in check gate |
| `background_fmt/` | FMT-profile-driven ISS background reads (real output for 3 samples as of 2026-09-21); `manifest.tsv` in the same schema `10_read_mixing.sh` expects, read by `10_read_mixing.sh` (repointed 2026-10-06) |
| `spike_in_samples/` | ART-based spike-in samples: `{type}/sample{n}/{condition}/final.{1,2}.fq.gz` — `og` design, 1 replicate, 20 conditions, built against `background_fmt/` (job 1123532, 2026-10-06); the old 5/10/20v `background_iss` output is archived at `archive/spike_in_samples_background_iss/` |
| `sample_lists/xad` | Sample IDs for the active detection pipeline run (one `{profile}_{model}` per line) |
| `sample_lists/xac` | FMT sample IDs for the **archived** contig-based classification (`mock_datasets/archive/kraken2_fmt.sh`, ERR2198611–613) |
| `sample_lists/xae` | FMT sample IDs for the **current** read-based classification (`FMT_READS_SAMPLE_LIST`, `mock_datasets/scripts/01_kraken2_fmt_reads.sh`, ERR2198703–705) |
| `accessions_in_kraken2.txt` | INPHARED spike-in genome pool — 408 accessions (of the full ~2902-genome INPHARED catalog) confirmed kraken2_pluspfp-detectable, written by `mock_datasets/scripts/08_check_kraken2_accessions.py` |

Assemblies (when produced) use SPAdes sequential naming (`k141_0`, `k141_1`, …) independently
per community — the same contig ID can appear in every assembly but refers to a different
sequence each time.

### Mock Dataset Outputs (`mock_datasets/results/`)

Separate from `agentic_benchmarking/results/` (the agentic detection workflow's own numbered stages) and from
`data/` (inputs/samples) — holds analysis output produced *while building* the FMT-derived
dataset itself:

| Directory | Contents |
|---|---|
| `kraken2_fmt/` | **Archived approach's** output — per-sample kraken2 classification of real FMT assemblies against `humgut_kraken2`, written by `mock_datasets/archive/kraken2_fmt.sh`. Kept, no longer the sample set going forward |
| `kraken2_fmt_reads/` | **Current** per-sample kraken2 classification of real FMT reads against `humgut_kraken2` — written by `mock_datasets/scripts/01_kraken2_fmt_reads.sh`, feeds Bracken correction |
| `representative_genomes.tsv` | One genome file per taxid (highest `prevalence_score`), written by `mock_datasets/scripts/04_pick_representative_genomes.py` |
| `fmt_abundance_profiles/` | Per-sample ISS `--abundance_file` TSVs, written by `mock_datasets/scripts/06_build_fmt_abundance_profile.py` from Bracken-corrected counts |
| `kraken2_check/` | Spike-in detectability gate output (`kraken2_out.txt`, `kraken2_report.txt`, `per_reference_recall.tsv`) — written by `mock_datasets/scripts/11_kraken2_check.sh`, moved out of the benchmark's `results/00_kraken2_check` since it's a dataset-QC step, not an agentic-workflow stage |

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
| `test_env` | BioPython/pandas helper scripts (`09_pick_refs.py`, `extract_humgut_subset.py`, `viral_spike_helper.py`, `combine_reads.py`), ART Illumina (viral spike-in reads), MegaHit |
| `insilicoseq` | InSilicoSeq (ISS) — background metagenome read simulation |

Each `@python_app` activates its specific environment via `conda run -n <env>` inside the function body.

## Key Files

| File | Purpose |
|---|---|
| `agentic_benchmarking/archive/agentic_viral_benchmark.py` | Original monolithic script — preserved, do not delete |
| `agentic_benchmarking/pipeline/coordinator.py` | Refactored entry point; orchestrates all agents |
| `agentic_benchmarking/pipeline/f1_eval.py` | F1/precision/recall; importable module and CLI |
| `agentic_benchmarking/pipeline/tool_selector.py` | Alpha-greedy tool selection |
| `config.py` | Python config — all paths, DB locations, parameters (single source of truth) |
| `agentic_benchmarking/scripts/run_viral_detection.sh` | SLURM submission wrapper — runs `agentic_benchmarking/pipeline/coordinator.py` |
| `mock_datasets/scripts/01_kraken2_fmt_reads.sh` | Classifies real FMT sample reads against the HumGut kraken2 DB (`config.HUMGUT_KRAKEN2_DB`) — feeds Bracken correction |
| `mock_datasets/scripts/02_bracken_build.sh` | One-time Bracken kmer-distribution index build for `HUMGUT_KRAKEN2_DB` |
| `mock_datasets/scripts/03_bracken_run.sh` | Runs Bracken per FMT sample to correct kraken2's ancestor-rank-collapsed read counts down to genome level |
| `mock_datasets/scripts/04_pick_representative_genomes.py` | Picks one genome file per taxid (highest `prevalence_score`) for taxids with multiple HumGut `cluster975` candidates |
| `mock_datasets/scripts/05_extract_humgut_phoraged.sh` (+ `helpers/extract_humgut_phoraged.py`) | Extracts only the needed phoraged HumGut genomes from the shared tarball shards |
| `mock_datasets/scripts/06_build_fmt_abundance_profile.py` | Builds the ISS `--abundance_file` per sample from Bracken-corrected counts |
| `mock_datasets/scripts/07_generate_background_fmt.sh` | Runs `iss generate --abundance_file` per FMT sample profile against the extracted phoraged genomes — novaseq only, 10M reads |
| `mock_datasets/scripts/08_check_kraken2_accessions.py` | Re-confirms kraken2_pluspfp detectability for the full ~2902-genome INPHARED catalog, writes `data/accessions_in_kraken2.txt` |
| `mock_datasets/scripts/09_pick_refs.py` | Picks INPHARED genomes (408-accession pool) and per-genome ART coverages per sample dir (`--design og`) |
| `mock_datasets/scripts/10_read_mixing.sh` | Builds every condition in each sample dir's `conditions.tsv` via ART, mixed with a random `data/background_fmt/` row |
| `mock_datasets/scripts/helpers/combine_reads.py` | Concatenates + shuffles viral spike-in reads with background reads |
| `mock_datasets/scripts/helpers/viral_spike_helper.py` | `log-actual-coverage`: derives per-genome coverage from ART's real read counts (QC log only) |
| `mock_datasets/scripts/11_kraken2_check.sh` | Gate: confirms spiked viral refs are kraken2-detectable before the heavier pipeline runs; output in `mock_datasets/results/kraken2_check/` |
| `mock_datasets/scripts/12_assemble.sh` | MegaHit assembly of each spike-in FASTQ pair |
| `mock_datasets/archive/submit_pipeline.sh` | Archived 2026-10-24 — chained the retired background step and percentage design |
| `mock_datasets/archive/generate_background.sh`, `mock_datasets/archive/extract_humgut_subset.py` | Superseded — built the old ISS-built-in-distribution background from raw HumGut MAGs |
| `mock_datasets/archive/kraken2_fmt.sh` | Superseded 2026-09-19 — contig-based FMT classification (ERR2198611–613); replaced by `mock_datasets/scripts/01_kraken2_fmt_reads.sh` |
| `mock_datasets/archive/bracken_build_fix.sh` | One-time patch for a `02_bracken_build.sh` bug (job 888905); folded back into `02_bracken_build.sh` 2026-09-22 |
| `mock_datasets/archive/remove_viral_contigs.py` | Superseded — built the old ART-era ERR-marine-metagenome background |
| `mock_datasets/archive/` (equal2/3/4, unequal2/3, single, check_kraken2_accessions.sh) | Retired sample-creation scripts (basis for `10_read_mixing.sh`'s ART coverage logic and `08_check_kraken2_accessions.py`) |
| `agentic_benchmarking/scripts/CheckV_parser.R` | R script for parsing CheckV output |

## Logs

All logs are written to `logs/` with three subdirectories:
- `logs/slurm/` — SLURM job stdout/stderr (from `agentic_benchmarking/scripts/run_viral_detection.sh` and the ISS sample-creation scripts)
- `logs/sample/` — per-sample processing logs from the old ART-era `mock_datasets/archive/create_single_equal_samples.sh`
- `logs/runinfo/` — Parsl workflow metadata (written automatically by the 4 Parsl configs)
