# Agentic Viral Benchmarking

Two pipelines with the same layout (`scripts/`, `archive/`, `results/`, plus a README each), sharing the root `config.py`:

| Pipeline | Status | What it does |
|---|---|---|
| [`mock_datasets/`](mock_datasets/README.md) | complete | Builds synthetic spike-in gut metagenomes. FMT-profile background (InSilicoSeq) + INPHARED phages (ART), assembled with MegaHit. Steps are numbered `01_` to `12_` in run order. |
| [`agentic_benchmarking/`](agentic_benchmarking/README.md) | in progress | Agentic, F1-driven benchmark of viral detection tools on those samples (`pipeline/`, Parsl + Academy). |

Shared at the root: `config.py` (all paths and parameters), `data/` (inputs and the spike-in samples, which are the hand-off between the two pipelines), and `logs/`.
