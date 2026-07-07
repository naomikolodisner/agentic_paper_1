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
- Added `scripts/kraken2_sanity_check.sh` as a first-step gate. Discovered AVrC references use
  custom catalog IDs (e.g. `GutCatV1_GPD_113896`), not NCBI accessions — confirmed zero overlap
  with `kraken2_pluspfp`'s `seqid2taxid.map`, so an ID-presence check (like the existing
  `check_kraken2_accessions.sh`) is meaningless here. The gate instead simulates reads per
  reference and runs one combined kraken2 classify call. Live-tested against 2 real AVrC
  references: only 1.4% of reads classified at all, one reference had **zero** classified reads
  — expected, since AVrC genomes are largely novel/uncultured metagenome-assembled viruses not
  represented in kraken2's general-purpose DB. This is a real, meaningful finding for the
  benchmark design, not a script bug.
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
