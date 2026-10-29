# Lab Notebook — Agentic Viral Benchmarking

A running, chronological log of decisions, debugging, and experiment results for this
project. Append new entries at the **bottom**, newest last. This complements `CLAUDE.md`
(which documents current architecture/how-to-run) — this file documents *why things changed*
and *what happened when we ran it*.

## Entry format

```
## YYYY-MM-DD

- What was done / tried
- What happened (results, errors, numbers)
- Decision / next step
```

---

## 2026-07-06

Baseline snapshot of in-progress (uncommitted) work, captured before starting today's session.

- `pipeline/coordinator.py`: `select_spike_in_samples()` changed from randomly picking a
  coverage level common across all sample dirs to picking the **highest** coverage
  (`max(..., key=...)`), and from running all sample dirs of the chosen type to running a
  **single randomly-chosen sample dir** per invocation. Also loosened `_resolve_viral_fasta()`
  to check the `.fa`/`.fna`/`.fasta` suffix without first requiring `os.path.isfile`, and split
  the F1-skip logging into two distinct cases (unresolvable FASTA vs. tool produced zero viral
  sequences).
- `pipeline/viral_detection.py`: VIBRANT invocation flag fixed from `--no_plot` to `-no_plot`
  (VIBRANT's actual CLI flag — the double-dash form was silently failing/being ignored).
- New script `scripts/check_kraken2_accessions.sh`: cross-checks an accession list (currently
  the VirFinder/INPHARED pre-2014 overlap list) against Kraken2's `seqid2taxid.map`, splitting
  into found/missing lists. Not yet wired into the main pipeline — exploratory/QC script for
  checking database coverage of reference accessions.
- Recent commit history (`Debugging` ×4, `Remove BUG file`) indicates an active debugging pass
  on the coordinator/tool-selection loop; no test suite currently exists to pin down regressions
  — verify pipeline behavior manually after changes land.

Tooling: set up this notebook to stay current without manual reminders.

- Added a "Lab Notebook" section to `CLAUDE.md` instructing proactive entries after any
  meaningful chunk of work (debugging pass, pipeline/config change, experiment run, or before
  a commit) — no need to ask each time.
- Added a `notebook` skill (`.claude/skills/notebook/SKILL.md`), invokable via `/notebook`, for
  on-demand entries reconstructed from git history + session context when a deliberate,
  well-formed entry is wanted (e.g. right before a commit).

## 2026-07-07

Switched synthetic sample creation from ART Illumina to InSilicoSeq (ISS), per notes from a
meeting relayed by the user. This touches `pick_refs.py`, `read_mixing.sh`, `assemble.sh`,
`submit_pipeline.sh`, `coordinator.py`, plus three new scripts, and changes the shape of the
spike-in dataset itself (not just the simulator).

- **Background source**: moving from the old ART/ERR-marine-metagenome background
  (`data/no_virus_contigs/`, now superseded but left in place) to ISS reads simulated from
  PHORAGER's prophage-stripped HumGut MAGs. That PHORAGER run **has not happened yet** — HumGut
  is only the downloaded tar at `/rs1/shares/brc/admin/databases/HumGut/HumGut2.tar`. Wrote
  `scripts/generate_background.sh` against the *expected* output path
  (`config.HUMGUT_PROPHAGE_REMOVED_DIR`, one FASTA file per MAG) so it's ready to run the moment
  that data lands; it fails fast with a clear message until then. Verified empirically that ISS's
  `--draft` treats each *file* passed to it as one genome — a single file concatenating many MAGs
  would be silently modeled as one organism, so the directory-of-one-file-per-MAG layout matters.
- **Spike-in scheme redesigned** after several rounds of clarifying with the user: dropped the
  old "amount of viruses" axis (`single`/`equal2`/`equal3`/`equal4`/`unequal2`/`unequal3`) x
  coverage (`0.1x`/`0.5x`/`1x`/`10x`) entirely. New scheme: each sample dir
  (`sample_{5,10,20}v_{1..5}/`) is a random subset of 5/10/20 AVrC viral genomes, spiked at a
  total read-fraction of 0.1%/1%/5%/10%, with the per-genome split inside a subset handled by
  ISS's own `--abundance` distribution (lognormal/halfnormal/exponential/uniform — all four
  generated as separate dataset variants, not just one default). Coverage is never an input —
  it's derived after the fact from ISS's own `*_abundance.txt` output + genome length + measured
  read length (`scripts/viral_spike_helper.py coverage-log`).
- Confirmed empirically (small live ISS runs, not just reading docs): `--n_reads` counts total
  reads across both mates (not pairs), `--abundance`/`--draft` proportions come back in
  `*_abundance.txt`, and HiSeq's actual read length is 126bp (not the 125bp the meeting notes
  quoted) — read length is measured from real output everywhere, never hardcoded, per the
  meeting's explicit warning about the old ART code's 100bp assumption.
- Added `scripts/kraken2_check.sh` (originally `kraken2_sanity_check.sh`, renamed 2026-07-07)
  as a first-step gate. Discovered AVrC references use custom catalog IDs (e.g.
  `GutCatV1_GPD_113896`), not NCBI accessions — confirmed zero overlap with `kraken2_pluspfp`'s
  `seqid2taxid.map`, so an ID-presence check (like the existing `check_kraken2_accessions.sh`)
  is meaningless here. The gate instead simulates reads per reference and runs one combined
  kraken2 classify call. Live-tested against 2 real AVrC references: only 1.4% of reads
  classified at all, one reference had **zero** classified reads — expected, since AVrC genomes
  are largely novel/uncultured metagenome-assembled viruses not represented in kraken2's
  general-purpose DB. This is a real, meaningful finding for the benchmark design, not a script
  bug.
- Every new script was smoke-tested end-to-end with small live runs (dummy MAGs, a real AVrC
  sample subset, tiny read counts) before being considered done: `generate_background.sh`
  (manifest write + idempotent skip), `viral_spike_helper.py` (both subcommands against known
  math), `read_mixing.sh` (all 16 dist x pct combos for one sample, correct read counts,
  idempotent re-run), `combine_reads.py` (found and fixed a quoting bug that broke process
  substitution for the background side), and `coordinator.py`'s rewritten
  `select_spike_in_samples()` (hand-built fake directory trees, including one with an
  intentionally incomplete tier).
- Deleted `scripts/gen_titration_sample.py` (superseded by `scripts/combine_reads.py`, which
  drops the old seqtk depth-subsampling logic entirely — no longer needed since both ISS outputs
  now have exact, pre-computed read counts and matching read length by construction).
- `config.sh` was discovered to be completely unused (no script sources it) and already stale
  (still on old `/xdisk/gwatts` paths, not even matching current `config.py`). Left it alone
  rather than maintaining a dead parallel file — `config.py` is the single source of truth.
- Not yet done: nothing can run end-to-end for real until PHORAGER produces the prophage-stripped
  HumGut MAGs. Next step once that lands: `bash scripts/submit_pipeline.sh`.
- Renamed `kraken2_sanity_check.sh` → `scripts/kraken2_check.sh` (updated `submit_pipeline.sh`,
  `CLAUDE.md`, `config.py` accordingly). While doing this, caught that the script hardcoded a
  lookup against the AVrC catalog FASTA to extract each reference's sequence — wrong going
  forward, since the planned dataset architecture (HumGut background + INPHARED spike-in) drops
  AVrC from spike-in generation entirely. Reworked extraction to pull each reference's sequence
  from whichever sample's own `contigs.fasta` already contains it (written by `pick_refs.py`
  regardless of source catalog), so the gate no longer depends on AVrC or any other specific
  catalog's FASTA/ID scheme.
- **Redesigned `kraken2_check.sh` again, more substantially**, after realizing the above still
  simulated a synthetic, background-free read set per reference rather than using the real
  spike-in reads that already exist once `read_mixing.sh` has run. Moved the gate in
  `submit_pipeline.sh` from before read-mixing to a new Step 1.5, between read-mixing and
  assembly, so it now runs on the real `final.1.fq.gz` files (background + virus already mixed
  by `combine_reads.py`) instead of a clean single-genome simulation.
  - Verified empirically (live `iss generate` test, not assumed from docs) that ISS's read
    headers embed the source genome ID as `<genome_id>_<counter>_<chunk>/1` (mate 2 ends `/2`),
    and confirmed `combine_reads.py`'s shuffle never rewrites headers — so real per-read genome
    provenance survives all the way through to the final mixed FASTQ. This is what makes it
    possible to check per-reference detectability using real data at all.
  - Considered two designs: (1) pre-filter each `final.1.fq.gz` down to just the known viral
    reads before classifying, vs (2) classify everything (background included) in one pool and
    filter kraken2's own output afterward by parsing the read-ID column. Went with (2) per
    explicit direction — kraken2 classifies each read independently regardless of what else is
    in the batch, so both give identical per-reference recall; (2) is simpler (no per-file awk
    extraction pass) at the cost of also classifying the much larger background read volume.
  - The report now distinguishes two failure modes that used to be conflated: a reference with
    zero *real* spike-in reads generated for it across every combo (a data-generation/sampling
    issue) vs. a reference with real reads that kraken2 simply never classified (the actual
    detectability signal this gate exists to catch).
  - `submit_pipeline.sh` job resources bumped (`--time` 4h→24h, `--mem` 32G→48G, `--cpus-per-task`
    8→16) since the check now classifies the full real dataset (background included) instead of
    a small fixed synthetic sample.

## 2026-07-15

Revised the background + spike-in generation design based on updated direction from the user
(dictated notes, several rounds of clarification needed — some contradictory on first pass,
resolved via `AskUserQuestion`). Two changes, scoped narrowly:

- **Background**: PHORAGER still hasn't run on HumGut (confirmed: `HumGut2.tar` has never even
  been *extracted* — only the 19GB tar + `HumGut2.tsv` manifest exist, ~31,226 genome rows). Per
  direction, background generation now runs on **raw** HumGut MAGs instead of waiting for
  PHORAGER. Added `scripts/extract_humgut_subset.py`: fixed-seed random sample of 500 genomes,
  extracted directly from the tar (idempotent, flock-guarded so `generate_background.sh`'s array
  tasks don't race on it) into `config.HUMGUT_RAW_MAGS_DIR`. Grid shrunk from 4 models × 4
  abundance dists × 3 depths (48 combos) to 4 models × 2 dists (`lognormal`, `exponential` —
  confirmed via `AskUserQuestion`, exponential representing the "few very abundant, most very
  low" skewed community) × 1 depth (1M total reads = 500k pairs, confirmed this is what "500,000
  reads, forward and reverse" meant) = **8 combos**. Depth list stays extensible for a shallower
  option later.
- **Spike-in reverted to ART Illumina** for the viral reads specifically (background pool stays
  ISS) — direction was to reuse the retired `scripts/archive/`
  equal2/equal3/equal4/unequal2/unequal3 scripts' approach rather than ISS's `--abundance`
  distribution. Final formula (took two follow-up questions to pin down, since "spiked at a
  total pct of reads" and "coverage alternates between the original ART options" initially read
  as contradictory): the pct-derived total viral read budget is still computed the same way as
  before, but is now split **unevenly** across a sample's genome subset by cycling the old
  scripts' fixed absolute coverages (`10, 1, 0.5, 0.1`) as *relative weights*
  (`config.ART_RATIO_WEIGHTS`), not split equally. Rewrote `scripts/viral_spike_helper.py`:
  `coverage-log` (ISS-abundance-file-based) replaced by `weighted-coverage-plan` (pre-ART,
  cycles the weights) and `log-actual-coverage` (post-ART, derives coverage from ART's real
  per-genome read counts — coverage stays a logged/derived quantity, never an input). Directory
  layout drops the `{dist}` level: `sample_{size}v_{n}/spike_{pct}pct/`.
  - **Caught a latent bug while porting the archived ART invocation**: the archived
    equal2/3/4/unequal2/3 scripts never passed `-p` (paired-end) to `art_illumina`, despite
    relying on paired `*1.fq`/`*2.fq` output — this ART build (verified via `art_illumina --help`)
    requires `-p` explicitly for paired output. Added it in the new `read_mixing.sh`.
  - **ART has no native NovaSeq quality profile** (confirmed via `--help`: only
    GA1/GA2/HS10/HS20/HS25/HSXn/HSXt/MinS/MSv1/MSv3/NS50 exist). Mapped `hiseq→HS25`,
    `miseq→MSv3`, `nextseq→NS50`, `novaseq→HS25` (fallback) in `config.ART_PROFILE_BY_MODEL`, with
    max-supported-length guards (`ART_PROFILE_MAX_LEN`) so `read_mixing.sh` skips (re-picks a
    different background row) rather than invoking ART with an incompatible read length. Real
    ISS-measured lengths per model aren't known yet since background has never been generated —
    flagged as something to check once `generate_background.sh` actually runs.
  - **Found and fixed a second, more serious bug this change would otherwise have introduced
    silently**: `scripts/kraken2_check.sh`'s per-reference attribution parses genome IDs back out
    of kraken2's read-ID column assuming ISS's naming convention
    (`<genome_id>_<counter>_<chunk>/1`). Verified empirically (live `art_illumina` test) that
    ART's actual convention is `<genome_id>-<read_number>/1` (hyphen, not underscore+counter) —
    the old regex would never have stripped ART's suffix correctly, silently reporting zero
    classified reads for every viral reference once spike-in reads became ART-generated. Fixed
    the regex and updated the surrounding comments/`CLAUDE.md` accordingly.
- Smoke-tested everything end-to-end before considering it done: `config.py` imports and all new
  variables resolve; `viral_spike_helper.py`'s new subcommands checked against hand-computed
  expected values; a full `read_mixing.sh` dry run (patched paths, fake background + 5-genome
  toy sample) produced correct weight-cycled coverage (genomes 1 & 5 both got weight 10 → highest
  read counts, genome 4 got weight 0.1 → lowest, matching the `[10,1,0.5,0.1]` cycle), sane final
  read counts scaling with pct (e.g. +370 pairs at 10% against a 3330-pair background, matching
  the `total-reads` formula almost exactly), and confirmed real read IDs in the final shuffled
  mix look like `GutCatV1_GPD_000005-149/1` — validating the `kraken2_check.sh` regex fix against
  real output, not just the docs. Also confirmed `extract_humgut_subset.py`'s tar-extraction path
  against the real `HumGut2.tar` (a 3-genome test took ~42s just for the sequential tar scan to
  reach those members — expected, tar has no index; the full 500-genome extraction will take
  longer but comfortably fits `generate_background.sh`'s 8h walltime).
- Updated `CLAUDE.md` (Synthetic Sample Creation section, config variable list, Key Files table,
  conda env table) and `scripts/submit_pipeline.sh` (array bounds: background 0-47→0-7, assembly
  0-239→0-59, dropped the now-false "PHORAGER must have run" prerequisite) to match.
- **First real (non-smoke-test) run, restricted to MiSeq only** as a quick end-to-end check
  before committing to the full 4-model grid: temporarily edited `MODELS=(miseq)` in
  `scripts/generate_background.sh` (marked `# TEMP`, needs reverting before a real run) and
  submitted the full chain (`generate_background.sh` → `read_mixing.sh` → `kraken2_check.sh` →
  `assemble.sh`) with SLURM `afterok` dependencies (jobs 459023–459027). `pick_refs.py` didn't
  need re-running at this point — the 15 sample dirs already existed from 7/7. Flagged the MSv3
  (MiSeq's ART profile) 250bp length cap as a real risk to check once the background manifest
  had real measured read lengths.
- **Switched the spike-in genome catalog from AVrC to INPHARED**, per direction: turns out the
  "103" genome count mentioned matches `data/accessions_in_kraken2.txt` exactly — the "found"
  output of the archived `check_kraken2_accessions.sh`, which cross-checked the 106-accession
  VirFinder/INPHARED pre-2014 overlap list against kraken2_pluspfp's `seqid2taxid.map` (103
  found, 3 missing: `NC_002670`, `NC_004587`, `NC_004821`). This was a QC/exploratory artifact,
  never previously wired into `pick_refs.py` (which was still drawing from AVrC's ~1M-row
  catalog). Found the actual sequences already sitting at
  `/rs1/shares/brc/databases/inphared/genomes_in_inphared_and_virfinder_pre2014/` (106 individual
  single-record FASTAs, one per accession).
  - Added `config.INPHARED_ACCESSIONS_LIST` / `INPHARED_GENOMES_DIR`. Rewrote `pick_refs.py` to
    read the accession list directly and pull each genome's own small FASTA file instead of
    indexing the huge AVrC catalog — simpler and no longer needs `pandas` at all.
  - Confirmed `config.DB_DIR`/`ANNOTATIONS` (same AVrC database) are still used independently by
    the BLAST annotation step in `pipeline/coordinator.py` — left those alone; only
    `AVRC_ALL_SEQUENCES`/`AVRC_METADATA_CSV` (pick_refs.py-only) became unused, kept with a
    comment rather than deleted.
  - Updated `kraken2_check.sh`'s expected-outcome note in `CLAUDE.md`: unlike AVrC (novel/uncultured,
    expected low kraken2 recall), these 103 accessions were *already confirmed* present in
    kraken2's DB, so high classification recall is now the expected result — a low-recall finding
    here would actually be a real signal worth digging into, not a shrug.
  - Cancelled the in-flight MiSeq-only test (459023–459027; background gen hadn't finished, so
    no manifest rows were written yet — cleaned up the resulting partial/incomplete ISS temp
    files under `data/background_iss/miseq_{lognormal,exponential}_1M/` before resubmitting) and
    re-ran `pick_refs.py` for real against the actual `data/spike_in_samples/sample_*v_*` dirs
    (verified: correct 5/10/20 ref counts per dir, matches a dry-run against a scratch copy done
    beforehand with the same seed). Resubmitted the same MiSeq-only chain fresh (jobs
    459093–459097) — HumGut's 500-genome raw subset had already been extracted by the first
    attempt, so that step is now just an idempotent skip.
- Not yet done: the MiSeq-only test chain (459093–459097) is in flight, not yet confirmed
  passing — in particular the MSv3 250bp read-length-cap risk noted above is still unverified
  against real ISS-measured MiSeq read length. Next: check `data/background_iss/manifest.tsv`
  once background gen finishes, then let read_mixing/kraken2_check/assemble run through; revert
  `MODELS=(miseq)` back to all 4 before a real (non-test) run.

## 2026-09-11

Started replacing the synthetic-distribution ISS background with one driven by a real FMT
sample's abundance profile, and cleaned up dataset structure that had drifted out of sync with
`CLAUDE.md`.

- **Archived the old background pipeline.** `scripts/generate_background.sh` and
  `scripts/extract_humgut_subset.py` → `scripts/archive/` (`remove_viral_contigs.py` was already
  there). `data/background_iss/` (the 4-combo synthetic grid, real output) was already sitting
  in `data/archive/` from a Sep 7 reorg — but that same reorg had also swept `spike_in_samples/`,
  `sample_lists/`, `accessions_in_kraken2.txt` into `data/archive/` by mistake, breaking the
  still-active INPHARED spike-in machinery. Restored those three back to `data/`.
  `set3_simulated_metagenomes/` (the old F1-eval ground truth) stayed archived deliberately — no
  longer used at all; the new FMT-derived datasets are the ground truth going forward.
  `config.py`'s now-archived vars (`HUMGUT_TAR`/`HUMGUT_RAW_MAGS_DIR`/`BACKGROUND_ISS_DIR`/
  `ISS_MODELS`/etc., plus `SPADES_DIR`/`CONTIG_IDENTITIES`) moved into a labeled "Archived"
  section rather than deleted, repointed at their real archived paths.
- **Finished the HumGut Kraken2 DB** (`data/humgut_kraken2/`, renamed from `data/kraken2/` at
  some point outside this session): the final `kraken2-build --build` step had been killed
  partway through on a login node (stale `taxo.k2d.tmp`, no `hash.k2d`/`opts.k2d`) — it's since
  completed (`hash.k2d` 23.4GB, verified readable via `kraken2-inspect`: k=35/l=31 defaults,
  6140 taxonomy nodes, 99.6% classified under Bacteria as expected for a HumGut-only/no-human
  build). Deleted the now-unneeded 64GB `HumGut975_library.fna` intermediate afterward (trivially
  regenerable from `fna/*.fna.gz` if ever needed again).
- **Expanded the INPHARED spike-in pool from 103 to 408 accessions.** The old pool was the
  106-accession VirFinder/INPHARED-pre-2014 overlap list (103 kraken2-confirmed); re-ran the same
  kraken2_pluspfp detectability check against the *full* current INPHARED catalog (~2902 genomes,
  `all_genomes.txt`) instead — new `scripts/check_kraken2_accessions.py` (supersedes the archived
  bash version, which would've been too slow at this scale) found 408 detectable. Needed because
  `VIRAL_SUBSET_SIZES` also changed: real per-sample viral diversity in the FMT data, after
  dereplicating/clustering ~900–1700 raw viral contigs down to distinct genomes, turned out to be
  ~80–150 — replaced the old `[5, 10, 20]` diversity-scaling grid entirely with `[80, 115, 150]`.
  103 accessions wouldn't have supported subsets that large with independent replicate draws.
  `INPHARED_GENOMES_DIR` now points straight at `/rs1/shares/brc/databases/inphared/GenomesDB`.
- **Reorganized `results/` vs. a new `dataset_creation/` directory.** `kraken2_fmt.sh` (the
  user's own new script, classifies real FMT contigs against `HUMGUT_KRAKEN2_DB` — done, run for
  3 samples: ERR2198611–613) and `kraken2_check.sh` had both landed under `results/00_*`, numbered
  as if they were early agentic-workflow stages. They're not — they're dataset-creation/QC
  outputs, not part of the tool-benchmarking pipeline. Moved both to a new top-level
  `dataset_creation/` (`kraken2_fmt/`, `kraken2_check/`), leaving `results/` to start fresh at
  `01_viral_detection` (unchanged) through `05_annotate` (was `06_annotate`, closing the gap).
  Updated `config.py` (`OUT_KRAKEN2_FMT`, `OUTPUT`) and the hardcoded `OUT_DIR` in
  `kraken2_check.sh` to match. Did **not** touch `data/` in this pass (agreed: dataset creation
  gets its own *results*, not its own *data* tree) or `results/test_virsorter_10x/` (a one-off
  tuning experiment, not a numbered stage).
- Brought `CLAUDE.md` back in sync with all of the above (Data table, new "Dataset Creation
  Outputs" section, Synthetic Sample Creation prose, Configuration variable list, Key Files).
- **Not yet done** (explicitly paused, mid-plan): the FMT abundance-profile computation from
  `kraken2_fmt.sh`'s classification output, extraction of the needed phoraged HumGut genomes,
  the actual ISS background-generation step from that profile, and repointing `read_mixing.sh` /
  `scripts/submit_pipeline.sh` at the new background instead of the archived one. `read_mixing.sh`
  currently cannot run end-to-end — it still looks for a background manifest at the now-archived
  `data/archive/background_iss/`. Also unresolved: `pipeline/coordinator.py` hardcodes its BLAST
  merge-step output to `results_testing/05B-05D_*` directly, bypassing `config.py`'s (already
  stale) `BLAST_HITS`/`OUTPUT` vars entirely — flagged to the user, not fixed.

## 2026-09-19

Picked up the FMT abundance-profile step, and pivoted the sample set + method after finding a
real accuracy problem in the first attempt.

- **Ran `scripts/kraken2_fmt.sh`** (contig-based classification of ERR2198611–613 against
  `HUMGUT_KRAKEN2_DB`, from the prior session) — completed cleanly: 86.0–96.1% of contigs
  classified per sample. Output confirmed at `dataset_creation/kraken2_fmt/<sample>/`.
- **Discovered ERR2198611–613 have no raw/cleaned reads on Hazel at all** — checked
  `input_dataset.yaml.gz` inside their SPAdes output, which points at
  `/xdisk/bhurwitz/.../07_contam_removal/ERR2198611_*.fastq.gz`, a University of Arizona `/xdisk`
  path that predates this project and isn't reachable from Hazel. Only the assembly survived the
  migration. This is *why* the contig-coverage-weighting approach existed in the first place, not
  a shortcut — there was no read-level alternative for this sample set.
  `fmt_viruses/03_contam_removal/` does have real reads, but for a **different** batch
  (ERR2198703–705), not 611–613 — caught this before wrongly assuming they were interchangeable.
- **Switched the pilot sample set to ERR2198703–705** (new `data/sample_lists/xae`), which do have
  their actual host(human)-decontaminated paired reads on Hazel (`bowtie2 --un-conc-gz` vs. CHM13,
  0.53% human alignment rate — done upstream by `fmt_viruses/03_contam_removal/run_remove_human.sh`).
  Added `scripts/kraken2_fmt_reads.sh` (new config vars: `FMT_READS_DIR`, `FMT_READS_SAMPLE_LIST`,
  `OUT_KRAKEN2_FMT_READS`) to classify these directly — no assembly-coverage proxy needed. Ran
  clean: 96.56–96.62% of read pairs classified per sample. `kraken2_fmt.sh`'s contig-based output
  for 611–613 was left in place (not deleted) — still valid, just superseded as the sample set
  going forward.
- **Verified the DB resolves to individual-genome level, not just species** (briefly worried
  otherwise after comparing one sample's ~3791 detected species-rank taxa against HumGut's 31225
  total genomes — wrong comparison, a single sample naturally won't contain every HumGut genome).
  Confirmed directly from the library FASTA headers (`>kraken:taxid|<taxid>|<seqid>`): every
  genome file carries its own unique taxid, and no two `S`-rank rows in a report share the same
  name. Resolution is fine.
- **Quantified a real LCA-collapse problem**: for ERR2198703, 32.2% of classified reads landed at
  an ancestor rank (genus/family/order/class/phylum) rather than being credited to a specific
  HumGut genome — expected in a DB this dense with closely related gut-genome strains, but not
  something to hand-wave; decided (with the user, after walking through what Bracken actually
  does mechanically) that a corrected abundance profile was worth the one-time build cost.
- **Built the Bracken index** (`scripts/bracken_build.sh`, job 888905, `-k 35 -l 100` per the
  user's spec — 100bp is the closest round number to these samples' real 101bp reads). This DB's
  on-disk layout didn't actually match what `bracken-build` expects (verified by reading the
  vendored script, not assumed) — `taxonomy/{nodes,names}.dmp` and `library/` didn't exist
  (the original concatenated library FASTA was deliberately deleted post-build to save 64GB), and
  no `seqid2taxid.map` was ever kept. Rebuilt all three:
  - `taxonomy/{nodes,names}.dmp` — symlinked to the existing top-level `GTDB_*.dmp` files.
  - `library/library.fna` — re-decompressed from `fna/*.fna.gz` into **one** concatenated file,
    not 31225 separate ones. Caught via `getconf ARG_MAX` (2097152 bytes) vs. a computed ~3.4MB
    of argv this DB's path length would produce across 31225 files — bracken-build passes every
    file under `library/` as a separate kraken2 argv entry, so leaving them as separate files
    would have failed immediately with "Argument list too long" after burning through the
    decompression time. One file avoids this entirely with no change to the actual classification
    (kraken2 works per-FASTA-record, not per-file). Decompression parallelized across 10 workers
    (file list split into per-worker chunks, each appending to its own part-file to avoid
    concurrent-write corruption, concatenated in order afterward) — dry-ran this logic on a
    7-genome subset in scratch first (correct multi-contig handling, correct concatenation order)
    before committing the real run to a 48h `long`-QOS slot.
  - `seqid2taxid.map` — parsed directly from the library's own `>kraken:taxid|<taxid>|<seqid>`
    headers (this DB's build convention).
  - Job completed in 41 min (95GB peak memory against a 100G request — cut it closer than
    expected), self-classified 10,459,180 library sequences (62.9 Gbp) at 100.00%. Verified all
    three real outputs exist: `database.kraken` (24GB), `database100mers.kraken` (935MB),
    `database100mers.kmer_distrib` (216KB, the actual per-sample correction table).
- **Not yet done**: running `bracken` itself against the 3 samples' `.report` files to get
  corrected per-genome abundance; the actual profile-building script (sum/normalize into an
  ISS `--abundance-file`); everything downstream that was already pending (phoraged genome
  extraction, ISS background generation, repointing `read_mixing.sh`). `data/humgut_kraken2/`
  grew to 125GB with `library.fna` (64GB) and `database.kraken` (24GB) now sitting alongside the
  original DB files — both are regenerable from `fna/*.fna.gz` if ever needed again (same as the
  original library.fna was), flagged to the user as a disk-cleanup candidate but not deleted
  without asking (7.7TB still free on `/rs1/researchers/b/blhurwit`, not urgent).

## 2026-09-20

Ran `bracken` against the 3 samples' reports — caught and fixed a real bug in yesterday's
Bracken index before trusting its output.

- **First run (job 896449) reported "Reads distributed: 0" for all 3 samples** — Bracken did
  nothing beyond thresholding; `.bracken` output was byte-for-byte the raw kraken2 direct counts
  with `added_reads` = 0 on every row. Not a "nothing to correct" result — verified by comparing
  `.bracken` directly against the raw `.report`'s S-level rows.
- **Root cause, traced through the actual intermediate files rather than guessed**: inspected
  `database100mers.kmer_distrib` and found every single row referenced the exact same
  "genome_taxid" (273997528 — not even in our real 4xxxxxx custom taxid range) with an identical
  total-kmer count, for all 5563 lines. Traced upstream to `database100mers.kraken`
  (`kmer2read_distr`'s own output) — same corruption there. Then checked
  `data/humgut_kraken2/database.kraken` (the real kraken2 self-classification, computed
  correctly) directly: its own sequence-ID column is the **entire** first header token,
  including the `kraken:taxid|N|` prefix (e.g. `kraken:taxid|4522367|HumGut_17319_1`) — that
  prefix-stripping only happens at `kraken2-build --add-to-library` time, not during a plain
  `kraken2` classify call, which was a wrong assumption on 2026-09-19.
  `seqid2taxid.map` had been built with the *stripped* ID (`HumGut_17319_1`) as the key, so every
  lookup `kmer2read_distr` made against it failed silently and it fell back to one bogus constant
  value for every one of the library's 10,459,180 sequences — silently corrupting the whole
  distribution table while still exiting 0 and looking superficially fine.
- **Fix** (`scripts/bracken_build_fix.sh`, job 896467, 16m48s): regenerated `seqid2taxid.map`
  using the full header token as the key (validated the sed regex against real headers first),
  then re-ran `bracken-build` — it explicitly skips recreating `database.kraken` if it already
  exists, so this did **not** redo the expensive 41-minute self-classification pass, only the
  broken `kmer2read_distr` + `generate_kmer_distribution.py` step. New
  `database100mers.kmer_distrib` grew from 216KB (1 fake genome) to 13.3MB (verified 4939 real
  distinct genome taxids in the table).
  - **Side finding while verifying this**: sampled 2000 of the 31225 genome *files* directly by
    their own FASTA headers and found only 1049 distinct taxids among them (up to 82 different
    genome files sharing one taxid) — HumGut's custom taxonomy clusters near-identical genomes
    under one representative taxid by design, it is **not** strictly one taxid per genome file.
    Revises the 2026-09-19 claim that this DB "resolves to individual-genome level" — that check
    only looked at one sample's *detected* taxa (never checked the full DB for duplicates across
    all 31225 files) and happened not to hit a shared-taxid case. Doesn't affect the seqid2taxid
    fix's correctness, but means the eventual abundance-profile script will need to pick a
    representative genome file per taxid when a cluster has more than one, not assume 1:1.
  - Re-ran `bracken` (job 896489) on the same 3 samples with the fixed index: **10,447,474 /
    10,447,476 previously-stuck reads redistributed for ERR2198703** (11,027,490/11,027,491 for
    ERR2198704; 11,982,511/11,982,513 for ERR2198705) — essentially complete recovery, only 1-2
    reads per sample genuinely undistributable. Spot-checked the corrected `.bracken` output:
    *Phocaeicola vulgatus* went from 511,206 direct reads to 5,040,049 corrected (~10x, a genus
    with heavy k-mer sharing in HumGut as expected); *Phocaeicola dorei* 93,003 -> 615,801 (~7x).
    Numbers are directionally sensible, not just nonzero.
- **Not yet done**: the profile-building script (sum `.bracken`'s `new_est_reads` per genome,
  normalize into an ISS `--abundance-file`) — now needs to account for the taxid-cluster finding
  above when picking which specific genome file represents a multi-genome taxid. Also still
  pending from before: phoraged genome extraction, ISS background generation, repointing
  `read_mixing.sh`. Disk cleanup of `data/humgut_kraken2/library.fna` (64GB) and `database.kraken`
  (24GB) still offered to the user, not yet actioned.

## 2026-09-21

Built the two remaining pieces between Bracken's corrected output and an ISS-usable abundance
profile.

- **Clarified the taxid/gtdb_tax_id relationship** while explaining the "representative genome"
  problem to the user: our kraken2 DB's taxid literally *is* HumGut's `gtdb_tax_id` column
  (GTDB species-level), not a separate mapping. Confirmed by checking `HumGut2.tsv` directly:
  31226 total rows (matches our 31225 genome files exactly -- HumGut's own catalog is already
  one row per `cluster975`, their finer ~97.5%-ANI strain cluster) vs. 4940 distinct
  `gtdb_tax_id` values (matches the 4939 distinct taxids seen in the fixed Bracken index). So
  several `cluster975`-level genome files legitimately share one taxid in our DB -- not a bug,
  a real consequence of using GTDB species as the taxonomy resolution.
- **Picked a selection criterion with the user**: discussed the standard dRep/GTDB convention
  (completeness − 5×contamination, from Olm et al. 2017) vs. `prevalence_score` (how consistently
  a specific genome was detected across HumGut's reference metagenomes) -- user chose
  `prevalence_score`, since the goal is a realistic background community, not the cleanest
  reference assembly.
- **`scripts/pick_representative_genomes.py`**: collects every taxid with nonzero corrected
  abundance across all 3 samples' `.bracken` files, picks the `HumGut2.tsv` row with the highest
  `prevalence_score` among that taxid's candidates (ties broken on `genome_file` for
  reproducibility). Ran clean, no missing-taxid warnings (confirms the taxid scheme
  understanding above). 3455 distinct taxids observed; 1932 (56%) had more than one candidate
  genome file. Spot-checked *P. vulgatus* (taxid 4507341, 16 candidates) by hand -- script's
  pick (`GUT_GENOME080427.fna.gz`, prevalence 0.9538) matches an independent manual sort.
  Output: `dataset_creation/representative_genomes.tsv`.
- **`scripts/build_fmt_abundance_profile.py`**: joins each sample's `.bracken` (`new_est_reads`)
  against the representative-genome table, sums per genome, normalizes to a fraction. One profile
  per sample (matches "a few replicate metagenomes" from the user's original dictated notes,
  rather than pooling all 3 into one). Verified: genome counts per profile (2335/3107/3137)
  exactly match Bracken's own "species with reads > threshold" counts; abundances sum to 1.0;
  top genome per profile matches Bracken's own `fraction_total_reads` column exactly.
  Output: `dataset_creation/fmt_abundance_profiles/<sample>_abundance.tsv`.
- **Important limitation flagged in the script itself, not glossed over**: this is *not* yet a
  real ISS `--abundance_file` usable as-is. Checked ISS's own source
  (`iss/generator.py`/`iss/abundance.py`) rather than trusting docs/memory: when
  `--abundance_file` is combined with `--draft`, ISS matches the abundance file's `genome_id`
  column against the *literal file path string* passed to `--draft` on the command line -- not a
  basename, not the FASTA record ID inside the file. `genome_id` here is just the bare HumGut
  accession (e.g. `GUT_GENOME080427`), since the real extracted-genome file paths don't exist yet
  (phoraged extraction step still not built). Whatever writes the eventual `--draft` list for
  `iss generate` will need to translate accession -> real path before this is directly usable.
- **Not yet done**: `scripts/extract_humgut_phoraged.py` (pull the ~3000-3500 needed genomes per
  sample out of the `HumGut_phoraged_*.tar.gz` shards at
  `/rs1/shares/brc/admin/databases/humgut_phoraged/`), the actual ISS background-generation step,
  repointing `read_mixing.sh`. Also still open: the `data/humgut_kraken2/library.fna` (64GB) /
  `database.kraken` (24GB) disk-cleanup offer from 2026-09-19, not yet actioned.

## 2026-09-21 (cont.)

Built the phoraged genome extraction step and closed the loop on the abundance profiles.

- **`scripts/extract_humgut_phoraged.py` + `scripts/extract_humgut_phoraged.sh`**: pulls only
  the genomes our profiles need out of the 7 shared `HumGut_phoraged_*.tar.gz` shards
  (`/rs1/shares/brc/admin/databases/humgut_phoraged/`) rather than fully unpacking all ~18GB.
  Uses `tarfile` stream mode (`r|gz`, single sequential pass per shard, extracting matches as
  encountered) instead of `getmembers()` + `extractfile()`, which would silently re-decompress
  the archive per lookup. Handled a naming mismatch: the archives key files by `HumGut_name`
  (e.g. `HumGut_2753_phoraged.fasta.gz`), but our representative-genome table uses the
  `genome_file` column -- joined through `HumGut2.tsv`.
  - Live-tested against the actual shards before committing to a full run (each shard took
    ~14s to stream through, far faster than expected) -- ran all 7 directly rather than via the
    prepared SLURM array wrapper, since total time was ~100s, well under what queueing through
    SLURM would cost in overhead alone. Output: `/rs1/researchers/b/blhurwit/users/nkolodi/
    databases/HumGut/prophage_removed/<genome_id>.fna` (one file per genome, config
    `HUMGUT_PROPHAGE_REMOVED_DIR`, reactivated from the "Archived" section of `config.py` since
    it's the same path from the old raw-MAG-subset approach, now genuinely in use).
- **Found PHORAGER hasn't processed all of HumGut**: summed member counts across all 7 shards
  (30691) against HumGut's real 31225 total -- 534 genomes (1.7%) were never phoraged at all, not
  an extraction bug. 26 of our 3455 needed genomes happened to fall in that gap (3429 extracted).
  Quantified the actual impact before deciding what to do: affected genomes represent
  0.00005-0.00007 of total abundance mass per sample (14-22 genomes out of ~2300-3100 per
  profile) -- negligible, so `build_fmt_abundance_profile.py` now filters the profile against
  what's actually available in `HUMGUT_PROPHAGE_REMOVED_DIR` and renormalizes, rather than
  falling back to an intact-prophage sequence for methodological consistency (dropped genomes
  and their exact abundance-mass cost are logged to stderr per sample, not silently discarded).
  Final profile sizes: ERR2198703 2321 genomes, ERR2198704 3086, ERR2198705 3115. All three still
  sum to 1.0 after renormalization.
- **Still not usable as a real ISS `--abundance_file`** for the same reason noted 2026-09-21
  earlier today: `genome_id` is the bare accession, not a real file path. Now that
  `HUMGUT_PROPHAGE_REMOVED_DIR` actually has files in it, though, the path resolution is trivial
  (`HUMGUT_PROPHAGE_REMOVED_DIR / f"{genome_id}.fna"`) -- just needs to be done by whatever script
  builds the `iss generate --draft` file list.
- **Not yet done**: the actual ISS background-generation script
  (`generate_background_fmt.sh`/equivalent) that reads these profiles, resolves genome_id -> real
  path, and calls `iss generate --draft ... --abundance_file ...` per sample; repointing
  `read_mixing.sh` at the new background instead of the archived one. Also still open: disk
  cleanup of `data/humgut_kraken2/library.fna` (64GB) / `database.kraken` (24GB).

## 2026-09-21 (cont. 2)

Closed the last gap in `build_fmt_abundance_profile.py`: now that
`scripts/extract_humgut_phoraged.py` has actually populated `HUMGUT_PROPHAGE_REMOVED_DIR`,
the script resolves each representative genome's bare accession into its real, absolute,
resolved file path (`(HUMGUT_PROPHAGE_REMOVED_DIR / f"{genome_id}.fna").resolve()`) and writes
that as the first column instead of the bare accession. Verified: top row of
`ERR2198703_abundance.tsv` is now `.../HumGut/prophage_removed/GUT_GENOME080427.fna    0.2104`,
and that exact path resolves to a real 2.4MB file on disk. Abundances still sum to 1.0 after the
change. This makes the profiles genuinely usable as a real ISS `--abundance_file` now, not just
correct-in-spirit -- documented in the script and in `config.py` that whatever calls
`iss generate --draft` should build its file list from this file's own first column directly
(e.g. `cut -f1`) rather than reconstructing paths independently, so the two can't silently drift
out of sync.

Remaining per the original plan: the actual `iss generate` background-simulation step, and
repointing `read_mixing.sh` at it.

## 2026-09-21 (cont. 3)

Built and ran the actual ISS background-generation step.

- **`scripts/generate_background_fmt.sh`**: per user direction, scoped to novaseq-only (dropped
  hiseq) at 10M total reads/5M pairs per combo (up from the old 1M-read default) -- reasoned
  through with the user first: our profiles have 2321-3115 genomes with a realistic long-tailed
  abundance distribution, vs. the old approach's 500 raw genomes, so 500k pairs would have
  averaged only ~150-215 pairs/genome and left most of the tail at single-digit/zero reads.
  Smoke-tested the actual `iss generate --draft ... --abundance_file ...` invocation at tiny
  scale (2000 reads) before committing to the real run -- confirmed correct paired output, and
  that ISS's own read naming embeds the true source genome ID, not the file path.
- **Submitted `--array=0-2`, mid-run got a "stop the jobs, I have an edit" interrupt from the
  user** -- cancelled (job 908615) before rerunning with the novaseq/10M change (job 908989).
- **Real bug caught in the new script, not ISS or the environment**: the read-length measurement
  line (`zcat R1.fastq.gz | awk 'NR==2{print length($0); exit}'`, copied from the archived
  generate_background.sh) relies on awk's early `exit` closing the pipe, which sends zcat SIGPIPE
  (exit 141). The archived script never hit this because it never set `pipefail`; this new script
  did (following this project's general convention), so all 3 array tasks FAILED at that exact
  line -- but only *after* `iss generate` itself had already completed successfully and written
  valid, complete R1/R2 FASTQ files (confirmed: ~5M read pairs each, matching the 10M-total-reads
  request). Fixed by toggling `pipefail` off around just that one command. Manually finished the
  bookkeeping (measured real read length/pair count, hand-appended manifest rows in the exact
  schema) for all 3 already-completed combos rather than wasting the ~45 min of ISS compute each
  had already spent by blindly rerunning.
- **Result**: `data/background_fmt/manifest.tsv` now has 3 real rows (ERR2198703/704/705, novaseq,
  151bp measured read length, ~5M pairs each), each with a real background FASTQ pair on disk at
  `data/background_fmt/<combo>/background_{R1,R2}.fastq.gz`.
- Wrote up a lab-meeting summary for the user covering everything from kraken2 classification
  through this step (not saved to a file -- delivered directly in conversation).

Next: user is separately editing `read_mixing.sh`'s viral spike-in side; once done, it needs its
`BACKGROUND_DIR`/`BACKGROUND_MANIFEST` repointed at `data/background_fmt/` (schema is a drop-in
match, no other changes needed on read_mixing.sh's side for this).

## 2026-09-22

Asked which scripts in `scripts/` are still active — surfaced that `kraken2_fmt.sh` (superseded
2026-09-19) and `bracken_build_fix.sh` (a one-off patch for `bracken_build.sh`'s job-888905 bug)
were both still sitting at the top level despite being fully superseded. Cleaned up both:

- **Moved `scripts/kraken2_fmt.sh` → `scripts/archive/`.** Nothing active referenced it directly
  (not called from `submit_pipeline.sh` or anywhere else); updated `config.py`'s
  `FMT_SAMPLE_LIST` comment to point at the new archived location.
- **Folded `bracken_build_fix.sh`'s fix into `bracken_build.sh` directly**, then moved
  `bracken_build_fix.sh` → `scripts/archive/`. The bug: `bracken_build.sh`'s `seqid2taxid.map`
  generation keyed on the *stripped* sequence ID (e.g. `HumGut_17319_1`), but kraken2 classify
  actually keeps the whole header token (`kraken:taxid|N|HumGut_17319_1`) as the sequence ID —
  stripping only happens at `kraken2-build --add-to-library` time, not classify time. That
  mismatch made every `kmer2read_distr --seqid2taxid` lookup fail silently. Rewrote
  `bracken_build.sh`'s `sed` pattern to key on the full header token from the start (matching
  what `bracken_build_fix.sh` already verified works), so a from-scratch run won't reproduce the
  bug and no longer needs a separate patch script. **Did not re-run anything** — the live
  `data/humgut_kraken2/` DB already has the corrected index from the fix script's earlier run
  (confirmed via `config.py`'s own comment: 4939 distinct genome taxids in the fixed index vs.
  4940 expected).
- Also patched several now-stale `CLAUDE.md` mentions of `kraken2_fmt.sh` as if it were still
  the active/current script (Running the Pipeline command list, Synthetic Sample Creation status
  paragraph, Data table, Dataset Creation Outputs table, Key Files table) — these hadn't been
  touched since 2026-09-11 and didn't reflect any of the Bracken/representative-genome/
  abundance-profile/phoraged-extraction work done 2026-09-19 through 2026-09-21. Added rows for
  the scripts that work covers (`kraken2_fmt_reads.sh`, `bracken_build.sh`, `bracken_run.sh`,
  `pick_representative_genomes.py`, `build_fmt_abundance_profile.py`,
  `extract_humgut_phoraged.py`/`.sh`, `generate_background_fmt.sh`) and flagged that
  `read_mixing.sh`/`submit_pipeline.sh` are still not repointed at `data/background_fmt/`.
  `CLAUDE.md` may still be incomplete beyond these specific corrections — a fuller pass wasn't
  in scope for this request.

## 2026-10-06

Resumed the spike-in step (the 2026-09-21 "Next" item).

- **Archived the old spike-in output.** `data/spike_in_samples/` only held the 5v/10v/20v dirs
  built against the archived `background_iss` (12 combos each) and the matching
  `assembly_work_list.txt`. Moved all of it to `data/archive/spike_in_samples_background_iss/`
  (3.0G). `read_mixing.sh` picks up every `sample_*v_*` dir, so leaving them in place would have
  rebuilt them against the new background.
- **Reran `scripts/pick_refs.py`** with the current `VIRAL_SUBSET_SIZES` `[80, 115, 150]` (seed
  41, 408-accession INPHARED pool). Created `sample_80v_1`, `sample_115v_1` and `sample_150v_1`;
  every accession resolved, with no missing-genome warnings. Also fixed the script's stale "103
  accessions" comment.
- **Repointed `scripts/read_mixing.sh`** at `data/background_fmt/` (`BACKGROUND_DIR`) and
  updated its header comment and error messages: 3 pct x 3 FMT backgrounds = 9 combos per dir,
  27 total. No logic changes were needed because the manifest schema matches.
- **Smoke-tested the coverage plan** for `sample_150v_1`, using 4,999,996 background pairs and
  151bp reads. At 1% (101,010 viral pairs), per-genome ART coverage ranges from 0.017x to 18.3x,
  median 0.41x. At 10% (1,111,110 pairs), it ranges from 0.18x to 201x, median 4.5x. So at 1%,
  most genomes in the 115v/150v sets are far below assemblable depth. This follows directly from
  the 10:1:0.5:0.1 weights spread over many genomes and is worth keeping in mind when reading
  recall.
- **Submitted `sbatch --array=0-2 scripts/read_mixing.sh` → job 1121376.**

Next: check job 1121376's output, then run `scripts/kraken2_check.sh`, then rebuild the assembly
work list and run `assemble.sh`. `submit_pipeline.sh` is still stale.

### 2026-10-06 (cont.) — switched spike-in from percentage-driven to plain equal coverage

- **Cancelled job 1121376** (the percentage-design run). It had already run 35–60 min per task
  and left 16 partial `spike_{1,5,10}pct_*` combo dirs across the 3 sample dirs. Those are left
  in place pending the user's decision; the new design writes to differently named dirs.
- **Why:** with the percentage design, coverage was a side effect of subset size. 150 genomes
  sharing 1% each got about half the coverage of 80 genomes sharing 1%, so diversity and depth
  were confounded. The cycled 10:1:0.5:0.1 weights also gave most 115v/150v genomes <1x at 1%.
  The user wants a "plain" spike-in like the original ART scripts instead.
- **`config.py`:** added `SPIKE_COVERAGES = [0.1, 0.5, 1, 10]`, the original
  `create_single_equal_samples.sh` grid. `SPIKE_PERCENTAGES` and `ART_RATIO_WEIGHTS` are marked
  UNUSED but kept, because `viral_spike_helper.py`'s `total-reads` and `weighted-coverage-plan`
  still reference them.
- **`scripts/read_mixing.sh` rewritten:**
  - Every genome is simulated at the same `-f {cov}` (`art_illumina -ss MSv3 -p -l 151 -m 200
    -s 10`, still one ART call per genome).
  - It builds every coverage x background row: 4 x 3 = 12 combos per sample dir, 36 total.
  - Output goes to `spike_{cov}x_{bg_combo}/`.
  - New `spike_summary.tsv` per combo logs the *derived* viral pair count and viral %.
    Expected viral %: 80v about 0.03/0.17/0.34/3.3%, 150v about 0.07/0.33/0.66/6.2%.
  - The previous version is backed up only in session scratch, not in the repo.
- **Smoke-tested** the per-genome ART call and the summary awk on 2 genomes at 1x (login node,
  light): pair counts are consistent with 1x, and read naming is preserved for
  `kraken2_check.sh`.
- **Also corrected:** an earlier note in this session suspected the percentage-design coverages
  were 2x too high. That was wrong; `viral_spike_helper.py`'s read-to-coverage math is correct.
- **Not submitted yet.** Waiting for the user's go-ahead.
- **Known downstream staleness, not fixed:** `pipeline/coordinator.py`'s
  `select_spike_in_samples()` parses a `{dist}/spike_{pct}pct/` layout, which matches neither
  the old nor the new naming.

### 2026-10-06 (cont. 2) — switched to an exact replica of the PI's original spike-in design

The PI wants the spike-in as close to the original ART scripts as possible. The user would
eventually like larger subsets but chose to replicate the original exactly for now. The
80/115/150 equal-coverage version was never submitted.

- **Deleted** the 17 partial `spike_*pct_*` dirs (11 GB) from cancelled job 1121376, with the
  user's OK. It was 17, not 16; one more combo finished before the cancel took effect.
- **Original design** (from `data/archive/{single,equal2,...}/`; 15 replicates per type; 300
  samples in total):

  | Type | Viruses | Conditions |
  |---|---|---|
  | single | 1 | 0.1 / 0.5 / 1 / 10x |
  | equal2, equal3, equal4 | 2, 3, 4 | 0.1 / 0.5 / 1 / 10x, same coverage for every genome |
  | unequal2 | 2 | `10x_1x`, `1x_0.5x` |
  | unequal3 | 3 | `10x_1x_0.5x`, `1x_0.5x_0.1x` |

  The original genomes were AVrC. The user chose fresh draws from the 408-accession INPHARED pool
  so the kraken2 gate stays meaningful, and one random FMT background per condition, as in the
  original (not all 3).
- **`config.py`:** added `SPIKE_OG_REPLICATES = 15` and `SPIKE_OG_DESIGNS` (above).
- **`scripts/pick_refs.py`:**
  - New `--design og` (default) writes `{type}/sample{1..15}/`; `--design subsets` keeps the
    80/115/150 layout.
  - Every sample dir now also gets a `conditions.tsv` (condition name, then per-genome coverages
    in `contigs.fasta` order).
  - A missing genome is now a hard error instead of a warning, since it would misalign the
    coverages.
  - Ran `--design og` (seed 41): 90 sample dirs, 225 picks, 176 unique genomes.
- **`scripts/read_mixing.sh`:**
  - Generic over `conditions.tsv`: one ART call per genome at that genome's own coverage.
  - One random usable background per condition.
  - Output goes to `{type}/sample{n}/{condition}/final.{1,2}.fq.gz`, plus `coverage_log.txt` and
    `spike_summary.tsv` (chosen background, viral pairs, derived viral %).
  - `conditions.tsv` is read on fd 3, so nothing in the loop can consume it from stdin.
  - Resources lowered to 2 CPU / 24G / 8h; the work is single-threaded ART, shuf and gzip.
- **`scripts/kraken2_check.sh`:** sample discovery now finds `conditions.tsv` dirs instead of the
  hardcoded `sample_*v_*` glob.
- **End-to-end test** (scratch copy, `unequal3/sample1`, 2,000-pair slice of the ERR2198703
  background, login node): both conditions built. Actual per-genome coverage was 9.997 / 0.990 /
  0.500x and 1.001 / 0.501 / 0.100x, matching the targets, and read counts reconcile.
- **Still the same as last entry:** MSv3 at 151 bp (forced by the novaseq background; the
  original was HS25 at 100 bp), and no subsampling to a fixed depth.
- The `sample_{80,115,150}v_1` dirs from earlier today have no `conditions.tsv`, so every script
  ignores them.
- **Not submitted.** Waiting for the user's go-ahead.
- **Cut to 1 replicate per type for initial testing** (user didn't want 90 sample dirs). Set
  `SPIKE_OG_REPLICATES = 1` (the original was 15), removed the 90 pick-only dirs (no reads in
  them), and reran `pick_refs.py --design og`. That gives 6 sample dirs (`{type}/sample1`) and
  20 conditions. Setting the value back to 15 and rerunning regenerates the full design.
- **Submitted the test as job 1123532** (`sbatch --array=0-5 scripts/read_mixing.sh`), with the user's go-ahead.
- **Job 1123532 result:** all 6 tasks COMPLETED (exit 0) in 20–40 min each, and all 20
  conditions have `final.{1,2}.fq.gz`. Output is 15 GB in total (about 750 MB per condition,
  more than the ~450 MB estimate).
  - **Coverage:** actual per-genome coverage matched every target, within 0.092–0.106x at 0.1x
    and within 1% elsewhere.
  - **Viral fraction:** derived viral % runs from 0.0003% (single 0.1x, 14 pairs) to 0.136%
    (equal4 10x, 6,800 pairs). All 3 backgrounds were drawn.
  - **Integrity**, spot-checked on `single/sample1/10x` and `unequal3/sample1/10x_1x_0.5x`:
    R1 and R2 pair counts are equal, totals match viral + background exactly, and the ART viral
    reads (`@NC_*`) are present in the shuffled output.

Next: `scripts/kraken2_check.sh` gate on these 20 samples, then the assembly work list and
`assemble.sh`.

## 2026-10-24

Cleanup pass, plus a workflow diagram of mock-dataset creation (Mermaid, for the Nextflow
integration project; kept outside the repo).

- **Deleted `data/spike_in_samples/sample_{80,115,150}v_1/`** (22 MB, picks only, no reads) at
  the user's direction. They were left over from the subsets design and had no `conditions.tsv`.
- **Archived `scripts/submit_pipeline.sh`** to `scripts/archive/`. It chained the retired
  `generate_background.sh` and percentage design, and the Nextflow port will replace job
  chaining. `assemble.sh` used to say the work list came from it, so its header and error
  message now give the one-line `find ... -name final.1.fq.gz | sort` command instead. Also
  reworded the matching comment in `kraken2_check.sh`.
- **`FMT_BACKGROUND_REPLICATES` 3 → 1.** Only rep1 was ever generated. The manifest's seeds
  (1000/1001/1002 for 703/704/705) are exactly what a 1-replicate run assigns, so the
  backgrounds were built at 1. Updated `generate_background_fmt.sh`'s header (`--array=0-2`).
- **The "kraken2_pluspfp" DB is really PlusPF, 16 GB capped, 2025-10-15**
  (`k2_pluspf_16_GB_20251015.tar.gz`; `library_report.tsv` has no plant library). Noted in
  CLAUDE.md; use that name in methods.
- **`check_kraken2_accessions.py` does no classification.** It only matches INPHARED accession
  IDs (`all_genomes.txt`) against `seqid2taxid.map`.
- **`viral_spike_helper.py` is QC only** in the og design. `log-actual-coverage` writes
  `coverage_log.txt`, which nothing downstream reads. `total-reads` / `weighted-coverage-plan`
  are unused (retired percentage design).
- **CLAUDE.md:** rewrote the dataset-prep run commands and the spike-in description for
  `--design og`, and updated the data and key-file tables.
- **Human-read removal of the 703–705 FMT reads** was done upstream by someone else. The
  bowtie2/CHM13 comments in `config.py` / `kraken2_fmt_reads.sh` were left as they are.

Next: `scripts/kraken2_check.sh` on the 20 og conditions (the `dataset_creation/kraken2_check/`
output is from 2026-07-21, old samples), then the assembly work list and `assemble.sh`.

## 2026-10-27

Reorganized the repo into two pipelines. Only code moved (`git mv`, history kept); no data moved.

- **`dataset_creation/`** is the mock-dataset pipeline (complete). All of `scripts/`, including
  `scripts/archive/`, is now `dataset_creation/scripts/`, next to the existing outputs
  (`kraken2_fmt_reads/`, `fmt_abundance_profiles/`, `kraken2_check/`, ...). `.gitignore` used to
  ignore all of `dataset_creation/`; it now ignores only the outputs and tracks `scripts/`.
- **`agentic_benchmarking/`** is the agentic pipeline (in progress). It holds `pipeline/`,
  `run_viral_detection.sh`, `archive/agentic_viral_benchmark.py` and `scripts/CheckV_parser.R`.
  Run it from `agentic_benchmarking/` (`python -m pipeline.coordinator`). The sbatch wrapper
  now `cd`s there.
- **Shared, left at the root:** `config.py`, `data/` (holds the spike-in samples, which are the
  output of dataset_creation and the input to the benchmark), `results/` (41G), `logs/`.
- **Path fixes:** every moved Python file's `sys.path` root went up one `.parent`, so
  `import config` still resolves. `$PROJECT/scripts/` became `$PROJECT/dataset_creation/scripts/`
  in the sbatch scripts. `config.CHECKV_PARSER` was repointed. Script mentions in comments,
  CLAUDE.md and README.md were updated. The archived `generate_background.sh` was already
  pointing at the wrong path for `extract_humgut_subset.py`; it now points at `archive/`.
- **Checked:** `bash -n` passes on every .sh. `--help` works for each dataset_creation Python
  script. `import pipeline.coordinator` works from `agentic_benchmarking/` in `academy_py311`.
  `CHECKV_PARSER` resolves. Nothing was re-run.

Next: the Nextflow port of dataset_creation (on hold for now).

Follow-up, same day: both pipelines now share one layout. This supersedes the
`dataset_creation/` paths in the entry above.

- **`dataset_creation/` was renamed `mock_datasets/`.** Both pipelines now use `scripts/`,
  `archive/`, `results/` and a `README.md`. `agentic_benchmarking/` also keeps its `pipeline/`
  package, and `run_viral_detection.sh` moved into its `scripts/`.
- **Data moved (same-filesystem renames, at the user's direction):** root `results/` →
  `agentic_benchmarking/results/` (`config.RESULTS_ROOT`). The dataset outputs (`kraken2_fmt*/`,
  `kraken2_check/`, `fmt_abundance_profiles/`, `representative_genomes.tsv`) went to
  `mock_datasets/results/` (`config.OUT_*`). No stored file referenced the old paths.
- **`mock_datasets/scripts/` holds only the steps that are actually run, numbered in run order:**
  `01_kraken2_fmt_reads.sh` … `07_generate_background_fmt.sh` build the background, and
  `08_check_kraken2_accessions.py` … `12_assemble.sh` build the spike-ins. Scripts that a step
  calls but that are never run on their own are in `scripts/helpers/` (`extract_humgut_phoraged.py`,
  `viral_spike_helper.py`, `combine_reads.py`). Retired scripts moved to `mock_datasets/archive/`
  (was `scripts/archive/`). The order, submit commands and outputs are in `mock_datasets/README.md`.
- The root README is now a short overview of the two pipelines. The old one became
  `agentic_benchmarking/README.md`. `.gitignore` dropped the `dataset_creation/*` special case,
  since the existing `results/` rule already ignores both `results/` dirs.
- **Checked:** `bash -n` passes on all .sh. `--help` works for each Python step and helper.
  Every Python file's `sys.path` root resolves to `config.py`. Every `$PROJECT/...` script path
  inside the sbatch scripts exists. All `config.OUT_*`, `RESULTS_ROOT` and `CHECKV_PARSER` paths
  exist. `import pipeline.coordinator` works. Nothing was re-run.

Dead-code cleanup, same day:

- **`viral_spike_helper.py`** now has only `log-actual-coverage`, the one subcommand
  `10_read_mixing.sh` calls. The retired percentage design's `total-reads` and
  `weighted-coverage-plan` were removed, along with the helper's `config` import, which only they
  needed. `config.SPIKE_PERCENTAGES` and `config.ART_RATIO_WEIGHTS` were removed too; only those
  subcommands read them. Checked by re-running the trimmed helper on
  `unequal3/sample1/10x_1x_0.5x`'s pair counts: the output is identical to the existing
  `coverage_log.txt`.
- **Deleted `config.sh`.** It was a stale shell copy of the config with pre-Hazel paths
  (`/xdisk/gwatts/...`, `/groups/gwatts/databases`) and nothing sourced it. It's still in git
  history.

## 2026-10-28

Nextflow port, kraken2 process (step 01):

- **`mock_datasets/nextflow/kraken2.nf`** now has a `kraken2` process that does what
  `01_kraken2_fmt_reads.sh` does: `kraken2 --paired --use-names` against the HumGut DB, writing
  `<sampleid>.kraken` + `<sampleid>.report` (same names as the sbatch script, so `03_bracken_run.sh`'s
  expectations carry over). Formatted to the hurwitzlab `nextflow_process_library` conventions
  (`modules/CLAUDE.md`): `process_high` + `publish_final` labels, `container
  "${params.container__kraken2}"`, `tuple val(sampleid), path(r1), path(r2)` + separate
  `path(kraken2_db)` input, single `emit: classification`, `shell:` with `!{}`, threads from
  `task.cpus`.
- The sbatch script's sample-list loop, config lookups and skip-if-report-exists check are left to
  the workflow (channel from `config.FMT_READS_SAMPLE_LIST` / `FMT_READS_DIR`; Nextflow `-resume`).
  Consuming pipeline must define `params.container__kraken2` and `params.kraken2_outdir`.
- Not run or linted yet.
- Simplified `01_kraken2_fmt_reads.sh` (~105 → 55 lines): cut the long header history (the
  contig-vs-reads backstory is already in CLAUDE.md and earlier entries), folded the four config
  lookups into one `cfg` helper, dropped the progress echoes and the out-of-range array check.
  Same kraken2 command and outputs. `bash -n` passes, not re-run.

## 2026-10-29

Nextflow port, kraken2 process (step 01), continued:

- **`kraken2.nf`** now uses conda, `conda "${params.conda__kraken2}"`, instead of
  `container "${params.container__kraken2}"`. The config should set it to the same env as
  `01_kraken2_fmt_reads.sh` (`/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/kraken2env`).
- No tuple. Inputs: `val(sample_id)`, `path(reads_dir)`, `path(kraken2_db)`. The process builds
  the read paths itself, as the sbatch script does: `<reads_dir>/<sample_id>/r1.fastq.gz` and
  `r2.fastq.gz` (one directory per sample, same file names in every sample). Outputs are two
  separate lines, `emit: report` and `emit: kraken` (replacing the single
  `emit: classification`). This fixes a half-done rename that had left
  `samplei_id`/`sampleid`/`sample_id` mixed. Flags are unchanged and match the sbatch script
  (`--paired --use-names`, `.kraken` + `.report`). The comment no longer says FMT/HumGut.
- Trade-off: `-resume` looks at `reads_dir`, not at each read file, so it may not rerun a sample
  whose reads were replaced in place. FMT reads (`03_contam_removal/<id>_1.fastq.gz`) aren't in
  this layout yet and would need per-sample dirs (symlinks are enough).
- No `main.nf`/`nextflow.config` yet (on purpose). The `process_high`/`publish_final` labels and
  `params.conda__kraken2` are still undefined.
- Not run or linted (no Nextflow on the laptop).
- **`01_kraken2_fmt_reads.sh`** now expects the same layout as the Nextflow process:
  `$FMT_READS_DIR/<sample>/r1.fastq.gz` and `r2.fastq.gz` (it used to be
  `$FMT_READS_DIR/<sample>_1.fastq.gz`). `config.FMT_READS_DIR` now points at `data/fmt_reads/`, which holds one
  dir per sample of `r1`/`r2` symlinks to the originals. The originals stay in
  `fmt_viruses/03_contam_removal` (now `config.FMT_READS_SOURCE_DIR`) and aren't modified. **The
  symlinks still have to be created on Hazel**, since `/rs1` isn't reachable from the laptop.
  CLAUDE.md data table updated. `bash -n` passes, not re-run.
