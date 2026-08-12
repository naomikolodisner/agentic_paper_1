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

**Dataset preparation (run before the main pipeline), run from project root:**
```bash
python scripts/pick_refs.py               # Pick a random viral-genome subset (5/10/20 refs) per sample
sbatch scripts/generate_background.sh     # Build the background pool from raw HumGut MAGs via ISS (4-combo array)
sbatch scripts/read_mixing.sh             # Build every percentage spike-in combo per sample via ART
sbatch scripts/kraken2_check.sh           # Gate: confirm spiked viral refs are kraken2-detectable in the real mixed reads
# then build the assembly work list and submit scripts/assemble.sh (array)

# or submit the whole chain with correct dependencies in one go:
bash scripts/submit_pipeline.sh
```
Background generation uses InSilicoSeq (ISS) — see "Synthetic Sample Creation" below. Viral
spike-in read simulation uses **ART Illumina** (reintroduced deliberately, scoped to spike-in
only — background stays on ISS), mixed with the ISS-generated background pool.
`scripts/remove_viral_contigs.py` and `data/no_virus_contigs/` / `data/background_assemblies/`
(the old ERR-marine-metagenome background) are superseded and no longer used by the current
scripts, but are left in place; `scripts/archive/` holds the retired
equal2/equal3/equal4/unequal2/unequal3-era sample-creation scripts that `read_mixing.sh`'s ART
coverage logic is modeled on.

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

- **Background**: `scripts/generate_background.sh` simulates the virus-free background
  metagenome from a random subset of **raw** HumGut MAGs (prophages intact — PHORAGER hasn't
  been run on HumGut yet; `scripts/extract_humgut_subset.py` extracts
  `config.HUMGUT_SUBSET_N` (500) genomes with a fixed seed straight from `config.HUMGUT_TAR`
  into `config.HUMGUT_RAW_MAGS_DIR`, since HumGut2.tar has never been extracted at all. Swap in
  `config.HUMGUT_PROPHAGE_REMOVED_DIR` once PHORAGER output lands). Each MAG file is passed as
  its own `--draft` argument — ISS treats each `--draft` *file* as one genome, so a single file
  concatenating many MAGs would wrongly be modeled as one organism (verified empirically
  2026-07-07). Builds a 4-combo grid (2 `config.ISS_MODELS` [`hiseq`, `novaseq`] x 2
  `BACKGROUND_ABUNDANCE_DISTS` [lognormal + exponential, a "typical" vs. skewed/"abnormal"
  community] x 1 `BACKGROUND_N_READS` [`1M`, i.e. 500k read pairs — the list is deliberately
  extensible, e.g. adding a shallower `"0.1M"` depth later just grows the grid]), logging each
  combo's **actual measured** read length and read-pair count to `config.BACKGROUND_MANIFEST` —
  read length is never hardcoded. `ISS_MODELS` is scoped to `hiseq`/`novaseq` only: ISS's
  `miseq`/`nextseq` models both measure ~301bp reads, which exceed every ART built-in profile's
  max supported length (ART's largest, MSv3, caps at 250bp), so ART can never simulate spike-in
  reads at a matching length for those two models — see `read_mixing.sh`'s `VALID_BG_ROWS`.
  `generate_background.sh` also prunes any manifest row that's permanently ART-incompatible
  (deleting its background files too), so a stale row from an earlier model choice can't linger
  forever and get selected by `read_mixing.sh`.
- **Spike-in**: `scripts/pick_refs.py` draws a random subset of `VIRAL_SUBSET_SIZES` (5/10/20)
  genomes per sample dir (`sample_{size}v_{n}/`) from the **INPHARED** catalog — specifically the
  103 accessions (`config.INPHARED_ACCESSIONS_LIST` → `data/accessions_in_kraken2.txt`) from the
  106-accession VirFinder/INPHARED pre-2014 overlap list confirmed present in kraken2_pluspfp's
  `seqid2taxid.map` (found by the archived `scripts/archive/check_kraken2_accessions.sh`; 3 of
  the 106 aren't in kraken2 and are excluded). Each accession is its own single-record FASTA in
  `config.INPHARED_GENOMES_DIR`. AVrC (`config.AVRC_ALL_SEQUENCES` / `AVRC_METADATA_CSV`) is no
  longer the spike-in source — those vars are unused now but kept since the same AVrC database
  (`config.DB_DIR` / `ANNOTATIONS`) still backs the unrelated BLAST annotation step in
  `pipeline/coordinator.py`. `scripts/read_mixing.sh` then
  builds **every** combination of `SPIKE_PERCENTAGES` (1%/5%/10% — 0.1% was dropped) x every
  background row an ART profile can actually simulate spike-in reads at (`VALID_BG_ROWS`,
  pre-filtered from `config.BACKGROUND_MANIFEST` by measured read length vs.
  `config.ART_PROFILE_BY_MODEL` / `ART_PROFILE_MAX_LEN`) — exhaustive, not a random per-combo
  draw. With the current 4-combo background grid that's 3 x 4 = 12 combos per sample dir, 36
  across all 3 sample dirs. For each combo it computes the total viral read target for that
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
  specifically chosen as the 103 accessions already confirmed present in `seqid2taxid.map`, so
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
- Set3 simulated assemblies: `SPADES_DIR` → `data/set3_simulated_metagenomes/assemblies/`
- Background assemblies (ERR* marine metagenomes, superseded): `BACKGROUND_ASSEMBLIES` → `data/background_assemblies/`
- Ground truth: `CONTIG_IDENTITIES` → `data/set3_simulated_metagenomes/contig_identities.csv`
- ISS background settings: `ISS_BIN`, `HUMGUT_TAR`, `HUMGUT_TSV`, `HUMGUT_SUBSET_N`,
  `HUMGUT_SUBSET_SEED`, `HUMGUT_RAW_MAGS_DIR`, `HUMGUT_PROPHAGE_REMOVED_DIR` (unused until
  PHORAGER runs), `BACKGROUND_ISS_DIR`, `BACKGROUND_MANIFEST`, `ISS_MODELS`,
  `BACKGROUND_ABUNDANCE_DISTS`, `BACKGROUND_N_READS`
- Spike-in settings (ART-based): `VIRAL_SUBSET_SIZES`, `SPIKE_PERCENTAGES`,
  `NUM_REPLICATES_PER_SUBSET_SIZE`, `ART_BIN`, `ART_FRAGMENT_MEAN`, `ART_FRAGMENT_SD`,
  `ART_RATIO_WEIGHTS`, `ART_PROFILE_BY_MODEL`, `ART_PROFILE_MAX_LEN`
- Spike-in genome catalog (INPHARED-based): `INPHARED_ACCESSIONS_LIST`, `INPHARED_GENOMES_DIR`
  (`AVRC_ALL_SEQUENCES` / `AVRC_METADATA_CSV` are unused by spike-in now, kept for `DB_DIR`/
  `ANNOTATIONS`'s unrelated BLAST-annotation use)
- Kraken2 check: `KRAKEN2_DB`, `KRAKEN2_SEQID2TAXID`

Helper functions: `init_dir()`, `create_dir()`, `lc()` (line count).

### Data (`data/`)

| Directory | Contents |
|---|---|
| `set3_simulated_metagenomes/assemblies/` | Per-sample `assembly.fa` files; `{profile}_{model}` subdirs (e.g. `SRR4831655_hiseq`) |
| `set3_simulated_metagenomes/contig_identities.csv` | Ground-truth viral identity; columns: `profile`, `model`, `query_id`, `superkingdom` |
| `background_assemblies/` | Original ERR\* marine metagenome assemblies used as background in the old ART pipeline (superseded) |
| `no_virus_contigs/` | Background assemblies with geNomad-identified viral contigs stripped out (superseded by ISS + HumGut/PHORAGER) |
| `background_iss/` | ISS-simulated background reads, one dir per model/abundance/n_reads combo; `manifest.tsv` logs each combo's actual read length + pair count |
| `spike_in_samples/` | ART-based spike-in samples mixed with the ISS background pool: `sample_{size}v_{n}/spike_{pct}pct_{bg_combo}/` |
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
| `scripts/pick_refs.py` | Picks a random viral-genome subset (from the 103-accession INPHARED catalog) per spike-in sample dir |
| `scripts/extract_humgut_subset.py` | One-time, idempotent extraction of a random raw HumGut MAG subset from `HumGut2.tar` |
| `scripts/generate_background.sh` | ISS background generation (model x abundance x depth grid) from raw HumGut MAGs |
| `scripts/viral_spike_helper.py` | Computes total viral read target from a percentage, splits it unevenly across a genome subset (`weighted-coverage-plan`), and derives per-genome coverage post-hoc from ART's real output (`log-actual-coverage`) |
| `scripts/read_mixing.sh` | Builds every percentage spike-in combo per sample via ART, mixed with the ISS background pool |
| `scripts/combine_reads.py` | Concatenates + shuffles viral spike-in reads with background reads |
| `scripts/assemble.sh` | MegaHit assembly of each spike-in FASTQ pair |
| `scripts/kraken2_check.sh` | Gate: confirms spiked viral refs are kraken2-detectable before the heavier pipeline runs |
| `scripts/submit_pipeline.sh` | Submits the full sample-creation chain with correct SLURM dependencies |
| `scripts/remove_viral_contigs.py` | Superseded — built the old ART-era ERR-marine-metagenome background |
| `scripts/archive/` | Retired equal2/equal3/equal4/unequal2/unequal3-era sample-creation scripts (basis for `read_mixing.sh`'s ART coverage logic) |
| `scripts/CheckV_parser.R` | R script for parsing CheckV output |
| `scripts/experimental/` | In-progress scripts not yet part of the main pipeline |

## Logs

All logs are written to `logs/` with three subdirectories:
- `logs/slurm/` — SLURM job stdout/stderr (from `run_viral_detection.sh` and the ISS sample-creation scripts)
- `logs/sample/` — per-sample processing logs from the old ART-era `scripts/archive/create_single_equal_samples.sh`
- `logs/runinfo/` — Parsl workflow metadata (written automatically by the 4 Parsl configs)
