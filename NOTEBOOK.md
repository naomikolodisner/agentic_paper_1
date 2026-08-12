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
