# Mock Datasets

Builds the synthetic spike-in metagenomes the agentic benchmark runs on: a virus-free gut
background simulated from real FMT abundance profiles (InSilicoSeq), with INPHARED phage genomes
spiked in at fixed coverages (ART Illumina), then assembled (MegaHit).

```
mock_datasets/
├── scripts/      the steps, numbered in run order (01 → 12)
│   └── helpers/  called by a step, never run directly
├── archive/      retired scripts (not part of the current pipeline)
└── results/      intermediate outputs (gitignored)
```

All paths and parameters come from the root `config.py`. Run every command from the project
root. The Python steps use the `test_env` conda env
(`/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python`).

## Run order

Steps 01–07 build the background and 08–12 build the spike-in samples. Step 08 depends on
nothing upstream, but 10 needs both 07 and 09.

| Step | Script | Submit | Output |
|---|---|---|---|
| 01 | `01_kraken2_fmt_reads.sh` | `sbatch --array=0-2` | `results/kraken2_fmt_reads/<sample>/<sample>.{kraken,report}` |
| 02 | `02_bracken_build.sh` | `sbatch` (one-time, already done) | Bracken index in `data/humgut_kraken2/` |
| 03 | `03_bracken_run.sh` | `sbatch --array=0-2` | `results/kraken2_fmt_reads/<sample>/<sample>.bracken` |
| 04 | `04_pick_representative_genomes.py` | `python` | `results/representative_genomes.tsv` |
| 05 | `05_extract_humgut_phoraged.sh` (→ `helpers/extract_humgut_phoraged.py`) | `sbatch --array=0-6` | phoraged genomes in `config.HUMGUT_PROPHAGE_REMOVED_DIR` |
| 06 | `06_build_fmt_abundance_profile.py` | `python` | `results/fmt_abundance_profiles/<sample>_abundance.tsv` |
| 07 | `07_generate_background_fmt.sh` | `sbatch --array=0-2` | `data/background_fmt/` + `manifest.tsv` |
| 08 | `08_check_kraken2_accessions.py` | `python` | `data/accessions_in_kraken2.txt` (spike-in genome pool) |
| 09 | `09_pick_refs.py --design og` | `python` | `data/spike_in_samples/{type}/sample{n}/` (`contigs.fasta`, `refs_log.txt`, `conditions.tsv`) |
| 10 | `10_read_mixing.sh` (→ `helpers/viral_spike_helper.py`, `helpers/combine_reads.py`) | `sbatch --array=0-$((N-1))`, N = number of `conditions.tsv` | `.../{condition}/final.{1,2}.fq.gz` |
| 11 | `11_kraken2_check.sh` | `sbatch` (gate: exits nonzero if a reference is undetectable) | `results/kraken2_check/per_reference_recall.tsv` |
| 12 | `12_assemble.sh` | `sbatch --array=0-$((N-1))`, N = lines in the work list | `.../{condition}/assembly.fa` |

The 0-2 arrays are one task per FMT sample in `data/sample_lists/xae`, and the 0-6 array is
one task per phoraged-HumGut shard. Every step skips work whose output already exists.

## Commands

```bash
PY=/rs1/researchers/b/blhurwit/users/nkolodi/conda_envs/test_env/bin/python

# Background (01-07)
sbatch --array=0-2 mock_datasets/scripts/01_kraken2_fmt_reads.sh
sbatch mock_datasets/scripts/02_bracken_build.sh                 # one-time
sbatch --array=0-2 mock_datasets/scripts/03_bracken_run.sh
$PY mock_datasets/scripts/04_pick_representative_genomes.py
sbatch --array=0-6 mock_datasets/scripts/05_extract_humgut_phoraged.sh
$PY mock_datasets/scripts/06_build_fmt_abundance_profile.py
sbatch --array=0-2 mock_datasets/scripts/07_generate_background_fmt.sh

# Spike-in samples (08-12)
$PY mock_datasets/scripts/08_check_kraken2_accessions.py
$PY mock_datasets/scripts/09_pick_refs.py --design og
N=$(find data/spike_in_samples -name conditions.tsv | wc -l)
sbatch --array=0-$((N - 1)) mock_datasets/scripts/10_read_mixing.sh
sbatch mock_datasets/scripts/11_kraken2_check.sh
find data/spike_in_samples -name final.1.fq.gz | sort > data/spike_in_samples/assembly_work_list.txt
N=$(wc -l < data/spike_in_samples/assembly_work_list.txt)
sbatch --array=0-$((N - 1)) mock_datasets/scripts/12_assemble.sh
```

The final samples in `data/spike_in_samples/` are the input to `agentic_benchmarking/`.
