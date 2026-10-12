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

**Dataset preparation (run before the main pipeline), run from project root. NOTE: this list is
known stale beyond `scripts/kraken2_fmt.sh` being archived — a whole FMT-profile background
sub-pipeline (Bracken correction, representative-genome selection, abundance profiles, phoraged
extraction, `generate_background_fmt.sh`) was built and has already run successfully
(`data/background_fmt/` has real output), but isn't fully documented here yet. See
`scripts/*.py`/`scripts/*.sh` docstrings and `config.py`'s "FMT-derived background" section for
the actual current step order in the meantime.**
```bash
python scripts/pick_refs.py               # Pick a random viral-genome subset (80/115/150 refs) per sample, from INPHARED
# scripts/kraken2_fmt.sh (contig-based FMT classification) is ARCHIVED -- superseded 2026-09-19
# by scripts/kraken2_fmt_reads.sh (read-based), followed by bracken_build.sh -> bracken_run.sh ->
# pick_representative_genomes.py -> build_fmt_abundance_profile.py -> extract_humgut_phoraged.sh
# -> generate_background_fmt.sh (already run for 3 samples: data/background_fmt/manifest.tsv)
sbatch --array=0-2 scripts/read_mixing.sh # repointed at data/background_fmt/ 2026-10-06
                                           # (run pick_refs.py first)
sbatch scripts/kraken2_check.sh           # Gate: confirm spiked viral refs are kraken2-detectable in the real mixed reads
# then build the assembly work list and submit scripts/assemble.sh (array)

# scripts/submit_pipeline.sh still chains the OLD (now-archived) generate_background.sh step —
# not yet updated for the FMT-profile background, don't rely on it until that's fixed.
bash scripts/submit_pipeline.sh
```
Viral spike-in read simulation uses **ART Illumina** (reintroduced deliberately, scoped to
spike-in only), mixed with a background pool. `scripts/archive/` holds retired sample-creation
scripts, including the original ISS-built-in-distribution `generate_background.sh` +
`extract_humgut_subset.py` (superseded by the FMT-profile approach) and the older
equal2/equal3/equal4/unequal2/unequal3-era scripts that `read_mixing.sh`'s ART coverage logic is
modeled on. `data/archive/` holds their old output.

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

### Synthetic Sample Creation

The background pool is built with [InSilicoSeq](https://insilicoseq.readthedocs.io/)
(`insilicoseq` conda env) for its realistic per-instrument error models (ISS ships pre-built
HiSeq/MiSeq/NextSeq/NovaSeq models; ART has no such per-instrument model and is
Illumina/PacBio-agnostic in a way that doesn't match real sequencers). Viral spike-in reads are
simulated with **ART Illumina** instead (`test_env` conda env) — reintroduced deliberately,
scoped to spike-in only, mirroring the retired `scripts/archive/`
equal2/equal3/equal4/unequal2/unequal3 scripts' fixed-coverage-per-genome approach — then mixed
with a randomly chosen background row from the ISS pool.

- **Background (being replaced — in progress)**: the original approach,
  `scripts/archive/generate_background.sh` (+ `scripts/archive/extract_humgut_subset.py`),
  simulated the virus-free background from a random subset of raw HumGut MAGs using ISS's
  built-in `lognormal`/`exponential` abundance distributions. **Archived** — output preserved at
  `data/archive/background_iss/`. It's being replaced with a background driven by a **real**
  microbiome abundance profile instead: classify an actual FMT sample assembly against a custom
  HumGut-only Kraken2 DB (`config.HUMGUT_KRAKEN2_DB` → `data/humgut_kraken2/`, GTDB taxonomy, no
  human library, built 2026-09-08), turn the classification into a per-genome abundance profile
  (`Σ(contig_length × SPAdes_coverage)` per HumGut genome, normalized), and simulate the
  background from that profile using PHORAGER's prophage-stripped ("phoraged") HumGut genomes
  instead of raw ones. **Current status (as of 2026-09-22, needs a fuller pass)**:
  `scripts/kraken2_fmt.sh` (contig-based classification, ERR2198611–613) is **archived** —
  superseded 2026-09-19 by read-based classification (`scripts/kraken2_fmt_reads.sh` against
  `data/sample_lists/xae`: ERR2198703–705, whose real decontaminated reads are on Hazel, unlike
  611–613 which only ever had a SPAdes assembly here). The full chain from there — Bracken
  correction (`scripts/bracken_build.sh` → `scripts/bracken_run.sh`, fixing kraken2's ~32%
  ancestor-rank-collapsed reads), `scripts/pick_representative_genomes.py`,
  `scripts/build_fmt_abundance_profile.py`, `scripts/extract_humgut_phoraged.sh`, and
  `scripts/generate_background_fmt.sh` — is built and **has already run successfully**:
  `data/background_fmt/manifest.tsv` has real novaseq/10M-read output for all 3 samples. `read_mixing.sh` was
  repointed at `data/background_fmt/` (`config.OUT_BACKGROUND_FMT`) 2026-10-06. Still broken:
  `submit_pipeline.sh` still chains the archived `generate_background.sh`.
  Any new background-generation script must still respect the same ISS constraint the old one
  did: each `--draft` *file* is treated as one genome, so a single file concatenating many
  genomes together would wrongly be modeled as one organism (verified empirically 2026-07-07).
- **Spike-in**: `scripts/pick_refs.py` draws a random subset of `VIRAL_SUBSET_SIZES`
  (`[80, 115, 150]`, matching the real ~80–150 distinct viruses found per FMT sample after
  dereplication — replaced the original `[5, 10, 20]` diversity-scaling grid) genomes per sample
  dir (`sample_{size}v_{n}/`) from the **INPHARED** catalog — specifically the 408 accessions
  (`config.INPHARED_ACCESSIONS_LIST` → `data/accessions_in_kraken2.txt`) of the full ~2902-genome
  current INPHARED catalog confirmed present in kraken2_pluspfp's `seqid2taxid.map` (found by
  `scripts/check_kraken2_accessions.py`, which supersedes the archived
  `scripts/archive/check_kraken2_accessions.sh` and its much smaller 103/106-accession
  VirFinder/INPHARED-pre-2014-overlap pool). Each accession is its own single-record FASTA in
  `config.INPHARED_GENOMES_DIR` (now `/rs1/shares/brc/databases/inphared/GenomesDB` directly).
  AVrC (`config.AVRC_ALL_SEQUENCES` / `AVRC_METADATA_CSV`) is no
  longer the spike-in source — those vars are unused now but kept since the same AVrC database
  (`config.DB_DIR` / `ANNOTATIONS`) still backs the unrelated BLAST annotation step in
  `pipeline/coordinator.py`. `scripts/read_mixing.sh` then
  builds **every** combination of `SPIKE_PERCENTAGES` (1%/5%/10% — 0.1% was dropped) x every
  background row an ART profile can actually simulate spike-in reads at (`VALID_BG_ROWS`,
  pre-filtered from `config.BACKGROUND_MANIFEST` by measured read length vs.
  `config.ART_PROFILE_BY_MODEL` / `ART_PROFILE_MAX_LEN`) — exhaustive, not a random per-combo
  draw (combo count depends on however many usable background rows exist once the FMT-profile
  background lands; the old 4-combo synthetic grid no longer applies). For each combo it computes the total viral read target for that
  percentage (`scripts/viral_spike_helper.py total-reads`), then
  splits that total **unevenly** across the subset's genomes by cycling
  `config.ART_RATIO_WEIGHTS` (`[10, 1, 0.5, 0.1]` — the old equal/unequal scripts' fixed
  absolute coverages, reused here as relative weights: `scripts/viral_spike_helper.py
  weighted-coverage-plan`) and converts each genome's share into an `art_illumina -f` fold
  coverage target via that genome's own real length. Each genome in the subset is then simulated
  individually with `art_illumina -ss {profile} -p -l {background's measured read length} -f
  {coverage_target} -m 200 -s 10` and concatenated into one viral R1/R2 pair. Per-genome coverage
  is a **derived, logged** quantity computed after the fact from ART's own real per-genome read
  counts (`scripts/viral_spike_helper.py log-actual-coverage`) — never an independent input.
  `scripts/combine_reads.py` concatenates + shuffles the viral spike with that background row into
  `sample_{size}v_{n}/spike_{pct}pct_{bg_combo}/final.{1,2}.fq.gz` — the background combo name is
  part of the directory now, since a given percentage is built against every usable background
  row rather than one randomly chosen one.
- **Gate**: `scripts/kraken2_check.sh` runs after read-mixing but before assembly, confirming
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
  no hardcoded catalog FASTA or ID scheme, just whatever `refs_log.txt` `pick_refs.py` already
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
- `scripts/assemble.sh` (MegaHit) is unchanged except for the FASTQ filename pattern.

### Configuration (`config.py`)

All paths, database locations, tool parameters, and output directory structure are centralized
here. `config.sh` (a shell mirror) exists but is stale and unused — no script sources it; treat
`config.py` as the single source of truth. Key variables:
- Project root: `/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1`
- Database root: `/rs1/researchers/b/blhurwit/users/nkolodi/databases`
- Tool root: `/rs1/researchers/b/blhurwit/users/nkolodi/tools`
- Input sample list: `XFILE_DIR / XFILE` → `data/sample_lists/xad` (sample IDs in `{profile}_{model}` format, e.g. `SRR4831655_hiseq`)
- **Archived** (moved to a clearly labeled section in `config.py`, kept so the archived scripts
  in `scripts/archive/` still import cleanly): `SPADES_DIR` / `CONTIG_IDENTITIES` (old Set3
  ground truth, repointed at `data/archive/set3_simulated_metagenomes/` — no longer used at all,
  the new FMT-derived datasets are the ground truth going forward), `HUMGUT_PROPHAGE_REMOVED_DIR`,
  `HUMGUT_TAR`, `HUMGUT_TSV`, `HUMGUT_SUBSET_N`, `HUMGUT_SUBSET_SEED`, `HUMGUT_RAW_MAGS_DIR`,
  `BACKGROUND_ISS_DIR` / `BACKGROUND_MANIFEST` (repointed at `data/archive/background_iss/`),
  `ISS_MODELS`, `BACKGROUND_ABUNDANCE_DISTS`, `BACKGROUND_N_READS`
- Background assemblies (ERR* marine metagenomes, superseded/archived): `BACKGROUND_ASSEMBLIES` → `data/archive/background_assemblies/`
- FMT-derived background (active, in progress — see "Synthetic Sample Creation"):
  `ISS_BIN`, `HUMGUT_KRAKEN2_DB` → `data/humgut_kraken2/`, `FMT_ASSEMBLIES_DIR`,
  `FMT_SAMPLE_LIST` → `data/sample_lists/xac`, `OUT_KRAKEN2_FMT` → `dataset_creation/kraken2_fmt/`
- Spike-in settings (ART-based): `VIRAL_SUBSET_SIZES` (`[80, 115, 150]`), `SPIKE_PERCENTAGES`,
  `NUM_REPLICATES_PER_SUBSET_SIZE`, `ART_BIN`, `ART_FRAGMENT_MEAN`, `ART_FRAGMENT_SD`,
  `ART_RATIO_WEIGHTS`, `ART_PROFILE_BY_MODEL`, `ART_PROFILE_MAX_LEN`
- Spike-in genome catalog (INPHARED-based, full ~2902-genome catalog): `INPHARED_ACCESSIONS_LIST`
  → `data/accessions_in_kraken2.txt` (408 confirmed), `INPHARED_GENOMES_DIR` →
  `/rs1/shares/brc/databases/inphared/GenomesDB` (`AVRC_ALL_SEQUENCES` / `AVRC_METADATA_CSV` are
  unused by spike-in now, kept for `DB_DIR`/`ANNOTATIONS`'s unrelated BLAST-annotation use)
- Kraken2 check gate (spike-in detectability, general-purpose DB): `KRAKEN2_DB`, `KRAKEN2_SEQID2TAXID`

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
| `humgut_kraken2/` | Custom HumGut-only Kraken2 DB (GTDB taxonomy, no human library), built 2026-09-08 + Bracken index (`scripts/bracken_build.sh`). Used by `scripts/kraken2_fmt_reads.sh` to classify real FMT reads for the FMT abundance profile — separate from `KRAKEN2_DB`/`kraken2_pluspfp`, which only backs the spike-in check gate |
| `background_fmt/` | FMT-profile-driven ISS background reads (real output for 3 samples as of 2026-09-21); `manifest.tsv` in the same schema `read_mixing.sh` expects, read by `read_mixing.sh` (repointed 2026-10-06) |
| `spike_in_samples/` | ART-based spike-in samples: `sample_{size}v_{n}/spike_{pct}pct_{bg_combo}/` — 80/115/150v built against `background_fmt/` (job 1121376, 2026-10-06); the old 5/10/20v `background_iss` output is archived at `archive/spike_in_samples_background_iss/` |
| `sample_lists/xad` | Sample IDs for the active detection pipeline run (one `{profile}_{model}` per line) |
| `sample_lists/xac` | FMT sample IDs for the **archived** contig-based classification (`scripts/archive/kraken2_fmt.sh`, ERR2198611–613) |
| `sample_lists/xae` | FMT sample IDs for the **current** read-based classification (`FMT_READS_SAMPLE_LIST`, `scripts/kraken2_fmt_reads.sh`, ERR2198703–705) |
| `accessions_in_kraken2.txt` | INPHARED spike-in genome pool — 408 accessions (of the full ~2902-genome INPHARED catalog) confirmed kraken2_pluspfp-detectable, written by `scripts/check_kraken2_accessions.py` |

Assemblies (when produced) use SPAdes sequential naming (`k141_0`, `k141_1`, …) independently
per community — the same contig ID can appear in every assembly but refers to a different
sequence each time.

### Dataset Creation Outputs (`dataset_creation/`)

Separate from `results/` (the agentic detection workflow's own numbered stages) and from
`data/` (inputs/samples) — holds analysis output produced *while building* the FMT-derived
dataset itself:

| Directory | Contents |
|---|---|
| `kraken2_fmt/` | **Archived approach's** output — per-sample kraken2 classification of real FMT assemblies against `humgut_kraken2`, written by `scripts/archive/kraken2_fmt.sh`. Kept, no longer the sample set going forward |
| `kraken2_fmt_reads/` | **Current** per-sample kraken2 classification of real FMT reads against `humgut_kraken2` — written by `scripts/kraken2_fmt_reads.sh`, feeds Bracken correction |
| `representative_genomes.tsv` | One genome file per taxid (highest `prevalence_score`), written by `scripts/pick_representative_genomes.py` |
| `fmt_abundance_profiles/` | Per-sample ISS `--abundance_file` TSVs, written by `scripts/build_fmt_abundance_profile.py` from Bracken-corrected counts |
| `kraken2_check/` | Spike-in detectability gate output (`kraken2_out.txt`, `kraken2_report.txt`, `per_reference_recall.tsv`) — written by `scripts/kraken2_check.sh`, moved here from `results/00_kraken2_check` since it's a dataset-QC step, not an agentic-workflow stage |

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
| `test_env` | BioPython/pandas helper scripts (`pick_refs.py`, `extract_humgut_subset.py`, `viral_spike_helper.py`, `combine_reads.py`), ART Illumina (viral spike-in reads), MegaHit |
| `insilicoseq` | InSilicoSeq (ISS) — background metagenome read simulation |

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
| `scripts/pick_refs.py` | Picks a random viral-genome subset (from the 408-accession INPHARED catalog) per spike-in sample dir |
| `scripts/check_kraken2_accessions.py` | Re-confirms kraken2_pluspfp detectability for the full ~2902-genome INPHARED catalog, writes `data/accessions_in_kraken2.txt` |
| `scripts/kraken2_fmt_reads.sh` | Classifies real FMT sample reads against the HumGut kraken2 DB (`config.HUMGUT_KRAKEN2_DB`) — feeds Bracken correction |
| `scripts/bracken_build.sh` | One-time Bracken kmer-distribution index build for `HUMGUT_KRAKEN2_DB` |
| `scripts/bracken_run.sh` | Runs Bracken per FMT sample to correct kraken2's ancestor-rank-collapsed read counts down to genome level |
| `scripts/pick_representative_genomes.py` | Picks one genome file per taxid (highest `prevalence_score`) for taxids with multiple HumGut `cluster975` candidates |
| `scripts/build_fmt_abundance_profile.py` | Builds the ISS `--abundance_file` per sample from Bracken-corrected counts |
| `scripts/extract_humgut_phoraged.py`/`.sh` | Extracts only the needed phoraged HumGut genomes from the shared tarball shards |
| `scripts/generate_background_fmt.sh` | Runs `iss generate --abundance_file` per FMT sample profile against the extracted phoraged genomes — novaseq only, 10M reads |
| `scripts/viral_spike_helper.py` | Computes total viral read target from a percentage, splits it unevenly across a genome subset (`weighted-coverage-plan`), and derives per-genome coverage post-hoc from ART's real output (`log-actual-coverage`) |
| `scripts/read_mixing.sh` | Builds every percentage spike-in combo per sample via ART, mixed with a background pool (reads `data/background_fmt/` since 2026-10-06) |
| `scripts/combine_reads.py` | Concatenates + shuffles viral spike-in reads with background reads |
| `scripts/assemble.sh` | MegaHit assembly of each spike-in FASTQ pair |
| `scripts/kraken2_check.sh` | Gate: confirms spiked viral refs are kraken2-detectable before the heavier pipeline runs; output in `dataset_creation/kraken2_check/` |
| `scripts/submit_pipeline.sh` | Submits the sample-creation chain with SLURM dependencies — **stale**, still calls the archived `generate_background.sh` |
| `scripts/archive/generate_background.sh`, `scripts/archive/extract_humgut_subset.py` | Superseded — built the old ISS-built-in-distribution background from raw HumGut MAGs |
| `scripts/archive/kraken2_fmt.sh` | Superseded 2026-09-19 — contig-based FMT classification (ERR2198611–613); replaced by `scripts/kraken2_fmt_reads.sh` |
| `scripts/archive/bracken_build_fix.sh` | One-time patch for a `bracken_build.sh` bug (job 888905); folded back into `bracken_build.sh` 2026-09-22 |
| `scripts/archive/remove_viral_contigs.py` | Superseded — built the old ART-era ERR-marine-metagenome background |
| `scripts/archive/` (equal2/3/4, unequal2/3, single, check_kraken2_accessions.sh) | Retired sample-creation scripts (basis for `read_mixing.sh`'s ART coverage logic and `check_kraken2_accessions.py`) |
| `scripts/CheckV_parser.R` | R script for parsing CheckV output |
| `scripts/experimental/` | In-progress scripts not yet part of the main pipeline |

## Logs

All logs are written to `logs/` with three subdirectories:
- `logs/slurm/` — SLURM job stdout/stderr (from `run_viral_detection.sh` and the ISS sample-creation scripts)
- `logs/sample/` — per-sample processing logs from the old ART-era `scripts/archive/create_single_equal_samples.sh`
- `logs/runinfo/` — Parsl workflow metadata (written automatically by the 4 Parsl configs)
