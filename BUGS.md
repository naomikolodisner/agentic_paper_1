# Bug Report: Agentic Viral Detection Pipeline

Generated: 2026-06-01

## Summary

The pipeline cannot run in its current state. There are 24 issues across 5 files. Fix them in the order listed below.

---

## Part 1: Critical Blockers

### `pipeline/parsl_configs.py`

**1. Invalid `cmd_timeout` parameter on three SlurmProvider calls (lines 66, 92, 118)**
`SlurmProvider` does not accept `cmd_timeout`. This raises `TypeError` at import time, before any pipeline code runs.

Fix: Remove `cmd_timeout=60*60*12` from `checkv_config`, `cmd_timeout=60` from `derep_cluster_config`, and `cmd_timeout=60` from `blast_config`.

**2. Conflicting walltime directives in `checkv_config` (lines 64, 67, 68)**
`scheduler_options='#SBATCH --time=12:00:00'` conflicts with `walltime='4:00:00'` and `SrunLauncher(overrides='--time=4:00:00')`.

Fix: Remove the `scheduler_options` time override and the `SrunLauncher` override; keep only `walltime='4:00:00'`.

---

### All four agent modules — Parsl lifecycle is broken

**3. All four agents call `parsl.clear()` / `parsl.load()` in `__init__`, destroying each other's config**
`ViralDetectionAgent.__init__` (viral_detection.py:290–291), `CheckVAgent.__init__` (checkv.py:98–99), `DereplicationClusteringAgent.__init__` (derep_cluster.py:90–91), and `BLASTAgent.__init__` (blast.py:154–155) each call `parsl.clear()` then `parsl.load()`. Each successive agent tears down the previous one's executor. After all four are initialized, only `blast_config` is active.

Fix: Remove `parsl.clear()`/`parsl.load()` from all agent constructors. Load a single combined Parsl `Config` once in `main()` before any agents are instantiated, using distinct executor labels (`viral_htex`, `checkv_htex`, `derep_htex`, `blast_htex`). Add the matching `executors=['viral_htex']` etc. to each `@python_app` decorator.

**4. All four agent subclasses skip `super().__init__()`**
`ViralDetectionAgent`, `CheckVAgent`, `DereplicationClusteringAgent`, and `BLASTAgent` all override `__init__` without calling `super().__init__()`. The Academy `Agent` base class initializes its action registry and internal state in `__init__`; skipping it leaves all `@action`-decorated methods unregistered.

Fix: Add `super().__init__()` as the first line of each agent's `__init__`.

---

### `pipeline/coordinator.py`

**5. Missing file: `solution1_manual.py` (line 305)**
`annotate_blast()` is called with `script_path = os.path.join(work_dir, 'solution1_manual.py')`. This file does not exist. Every round that reaches BLAST annotation raises `FileNotFoundError`.

Fix: Locate or create the annotation script and update the path, or implement the annotation logic inline.

**6. Double-await on coroutines (lines 150–153, 272–275, 284–290)**
`derep_fasta = await (await cluster_handle.run_dereplicate(...))` — the inner `await` already resolves to a string; the outer `await` then attempts to await a plain string, raising `TypeError`.

Fix: Remove the outer `await`; use a single `await` for each call.

**7. Tool selection permanently excludes 7 of 11 tools (line 300)**
`self.selector.choose_tool(['VirSorter2', 'DeepVirFinder', 'GeNomad', 'MARVEL'])` — VirSorter (v1), VirFinder, VIBRANT, viralVerify, ViraMiner, MetaPhinder, and Seeker can never be selected after round 1.

Fix:
```python
self.selector.choose_tool(ToolSelector.TOOLS)
```

**8. ZeroDivisionError in final round statistics (lines 316–318)**
`sum(self.match_ratios_history) / len(self.match_ratios_history)` raises `ZeroDivisionError` if all 10 rounds skipped BLAST. Same risk for `f1_history`.

Fix:
```python
final_avg_f1 = sum(self.f1_history) / len(self.f1_history) if self.f1_history else 0.0
```
Apply the same guard to `quality_ratios_history` and `match_ratios_history`.

**9. `CoordinatorAgent` launched with wrong API pattern (lines 347–350)**
All other agents are launched as pre-instantiated objects (`manager.launch(ViralDetectionAgent())`), but `CoordinatorAgent` is launched as a class with `args=`.

Fix:
```python
coord = CoordinatorAgent(viral_handle, checkv_handle, cluster_handle, blast_handle, shutdown_event)
await manager.launch(coord)
```

---

### `pipeline/viral_detection.py`

**10. Wrong hardcoded output filename for DeepVirFinder (line 68)**
`dvf_output = os.path.join(dvf_output_dir, 'contigs.fasta_gt1500bp_dvfpred.txt')`. DeepVirFinder names its output after the input FASTA: for `assembly.fa` the actual file is `assembly.fa_gt1500bp_dvfpred.txt`.

Fix:
```python
dvf_output = os.path.join(dvf_output_dir, os.path.basename(unzipped_spades) + '_gt1500bp_dvfpred.txt')
```

**11. Wrong hardcoded output path for geNomad (line 97)**
Returns `os.path.join(genomad_output_dir, 'contigs_summary', 'contigs_virus.fna')`. geNomad names its output directory and files after the input FASTA stem; for `assembly.fa` the output is `assembly_summary/assembly_virus.fna`.

Fix:
```python
stem = os.path.splitext(os.path.basename(unzipped_spades))[0]
return os.path.join(genomad_output_dir, stem + '_summary', stem + '_virus.fna')
```

**12. Wrong return path for MetaPhinder — copy-pasted from geNomad (line 238)**
Returns `os.path.join(metaphinder_output_dir, 'contigs_summary', 'contigs_virus.fna')`. MetaPhinder produces a tab-separated predictions file directly in the output directory, not a FASTA in a `contigs_summary/` subdirectory.

Fix: Return the actual MetaPhinder output file path. If a FASTA is needed, add extraction logic analogous to the DeepVirFinder pattern.

**13. `viraminer_app`: custom `env` dict created but never passed to `subprocess.run` (lines 211–218)**
`env['CUDA_VISIBLE_DEVICES'] = ''` is set to prevent GPU use, but `subprocess.run(cmd, stdout=f, check=True)` does not pass `env=env`.

Fix:
```python
subprocess.run(cmd, stdout=f, check=True, env=env)
```

**14. `viraminer_app`: `conda run` captures stdout, silently dropping the output file redirect**
Fix: Add `--no-capture-output` to the conda run command:
```python
cmd = ['conda', 'run', '--no-capture-output', '-n', 'viraminer', 'python', ...]
```

**15. `os.chdir()` called inside `@python_app` functions (lines 58, 67, 70, 79)**
Changes the global process CWD on a long-lived Parsl worker, corrupting subsequent tasks on the same worker.

Fix: Remove all `os.chdir()` calls; pass `cwd=` to each `subprocess.run()` call instead.

**16. `virfinder_app` uses hardcoded `~/.conda/...` Rscript path (line 136)**
On HPC compute nodes, `~` resolves to a service account home that likely doesn't have this conda env.

Fix: Replace with `conda run -n virfinder Rscript -e ...` via subprocess.

**17. `vibrant_app` passes `-no_plot` (single dash) instead of `--no_plot` (line 162)**
Python-based argument parsers treat `-no_plot` as concatenated short flags.

Fix: Change to `'--no_plot'`.

---

### `pipeline/checkv.py`

**18. Wrong conda environment name: `checkv` should be `checkv_env` (line 43)**
CLAUDE.md documents the CheckV environment as `checkv_env`, but the command uses `-n checkv`.

Fix: Change `'checkv'` to `'checkv_env'` on line 43.

**19. Undocumented conda environment `r_env` for Rscript (line 47)**
No environment named `r_env` is listed in CLAUDE.md.

Fix: Run `conda env list` on the cluster and use the correct environment name.

**20. `seqtk subseq` receives a CSV with a header row, not a plain ID list (lines 53–68)**
`cleaned_selection_csv` retains the `contig_id` header line. `seqtk subseq` looks for a sequence named `contig_id` and returns empty or wrong output.

Fix: Skip the header line when writing `cleaned_selection_csv`:
```python
if line.startswith('contig_id'):
    continue
```

**21. `PARSE_LENGTH` is int but subprocess args must be strings (line 48)**
Fix: Change `'-l', parse_length` to `'-l', str(parse_length)`.

**22. `os.chdir(work_dir)` inside `checkv_app` (line 70)**
Same CWD contamination problem as issue #15.

Fix: Remove the call; all paths are already absolute.

---

### `pipeline/derep_cluster.py`

**23. Wrong return value from `dereplicate_app` (line 45)**
Returns `os.path.join(out_derep, 'dereplicated.fasta')`, but no code ever writes a file with that name. The actual output is `cleaned_clusterRes_all_seqs.fasta`.

Fix: Change the return to `return cleaned_fasta`.

**24. `cluster_app` synchronization loop deadlocks the worker (lines 61–62)**
A `while not all(os.path.exists(flag) for flag in done_flags)` / `sleep` loop runs inside a `@python_app`, permanently blocking a Parsl worker slot with no timeout.

Fix: Remove the polling loop. The coordinator's `asyncio.gather` already ensures all `run_dereplicate` futures complete before `run_cluster` is called — no done-flag mechanism is needed.

---

### `pipeline/blast.py`

**25. Wrong BLAST results path in `merge_blast_results_app` (line 122)**
`run_blast_app` writes to `work_dir/results_testing/05C_blast`, but `merge_blast_results_app` reads from `os.path.join(work_dir, 'results', '05C_blast', ...)` — using `results` instead of `results_testing`.

Fix: Pass `blast_results_dir` as an explicit parameter to `merge_blast_results_app` and use it instead of reconstructing the path internally.

---

### `config.py`

**26. Typo in VirSorter output directory name (line 89)**
`OUT_VIRSORTER = RESULTS_ROOT / "01_viral_detection" / "01K_visorter"` — missing the `r` in `virsorter`.

Fix:
```python
OUT_VIRSORTER = RESULTS_ROOT / "01_viral_detection" / "01K_virsorter"
```

**27. Hardcoded user-specific BLAST path (line 80)**
`BLAST_PATH = "/home/u3/kolodisner/.conda/envs/blast/bin/"` — will fail for any other user or environment.

Fix:
```python
BLAST_PATH = os.environ.get('BLAST_BIN_PATH', '/home/u3/kolodisner/.conda/envs/blast/bin/')
```

---

## Part 2: Medium Issues

| File | Issue | Fix |
|---|---|---|
| `parsl_configs.py` | All four configs share executor label `Parsl_htex` — combining into one Config (required for blocker #3) will fail on duplicate labels | Rename to `viral_htex`, `checkv_htex`, `derep_htex`, `blast_htex` |
| `parsl_configs.py` | `_LOG_DIR` and `_PROJECT_ROOT` hardcoded to `/xdisk/gwatts/...` but repo is at `/rs1/researchers/...` | Source from `config.py` or ensure `/xdisk` is mounted before running |
| `viral_detection.py` | `virfinder_app` invokes `seqkit` without a conda environment (line 143) | Wrap with `conda run -n <seqkit_env> seqkit grep ...` |
| `coordinator.py` | No guard on empty `sample_ids` list (line 197) | `if not sample_ids: raise RuntimeError(...)` |
| `config.py` | `XFILE = "xad"` but `config.sh` uses `XFILE=xac` — Python and shell process different sample lists silently | Align the two files |
| `config.sh` | `CHECKV_PARSER` path missing `scripts/` subdirectory | Fix to `$PROJECT_ROOT/scripts/CheckV_parser.R` |

---

## Part 3: Unimplemented Stubs

**`_resolve_viral_fasta()` in `coordinator.py`**
Returns `None` for VIBRANT, viralVerify, and ViraMiner because their outputs are directories or non-FASTA files. F1 evaluation is always skipped, so these tools always score 0.0 and are permanently biased against in selection.

Fix: Implement FASTA extraction for each:
- VIBRANT: glob for `VIBRANT_phages_*/VIBRANT_phages_*.fna` inside the output directory
- viralVerify: glob for `*_Viral.fasta` inside the output directory
- ViraMiner: parse the predictions TXT, filter by score threshold, extract matching sequences from input FASTA

Until implemented, exclude these three tools from `choose_tool()`.

**`scripts/experimental/` — all three scripts are non-functional stubs**
- `agentic_spike.py`: hardcoded placeholder paths, ungenerated viral FASTQs, semantically broken F1 comparison (AVrC IDs vs SPAdes contig names), agent loop runs at import time
- `select_avrc_viruses.py` and `select_avrc_viruses_adaptive.py`: both import `avrc` — a package not installed in any documented environment

---

## Part 4: Systemic Issues

1. **Parsl lifecycle management is wrong throughout.** Every agent independently owns a Parsl DFK. This is architecturally incompatible with running multiple agents in the same process. Requires a single `Config` with multiple named executors loaded once in `main()`.

2. **Conda environment names are undocumented and inconsistent.** Run `conda env list` and check every `-n` argument in `pipeline/` against the actual available environments.

3. **Tool output paths are copy-pasted incorrectly.** The geNomad output path pattern was copied verbatim to MetaPhinder (and possibly others). Every tool's return path should be verified against the tool's actual documentation.

4. **`os.chdir()` used inside `@python_app` functions.** Parsl workers are long-lived; `os.chdir()` corrupts subsequent tasks on the same worker. Replace all instances with `cwd=` arguments to `subprocess.run()`.

---

## Recommended Fix Order

1. `parsl_configs.py` — remove invalid `cmd_timeout` (blocker #1). Cannot even import without this.
2. All agents — fix Parsl lifecycle (blockers #3, #4): remove `parsl.clear()`/`parsl.load()` from constructors, add `super().__init__()`, load single combined Config in `main()` with unique executor labels.
3. `derep_cluster.py` — fix `dereplicate_app` return value and remove deadlocking poll loop (blockers #23, #24).
4. `viral_detection.py` — fix tool output paths for DeepVirFinder, geNomad, MetaPhinder (blockers #10, #11, #12).
5. `viral_detection.py` — fix `viraminer_app` env and stdout issues (blockers #13, #14).
6. `checkv.py` — fix conda env name, seqtk header bug, `os.chdir` (blockers #18, #20, #22).
7. `coordinator.py` — fix double-await (blocker #6).
8. `coordinator.py` — create or locate `solution1_manual.py` (blocker #5).
9. `blast.py` — fix `merge_blast_results_app` path (blocker #25).
10. `coordinator.py` — fix tool selection list and implement `_resolve_viral_fasta` for VIBRANT/viralVerify/ViraMiner (blockers #7, stubs).
11. Verify all conda environment names against `conda env list`.
12. Fix medium issues and config mismatches.
