import os
import asyncio
import parsl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from glob import glob
from pathlib import Path

from academy.manager import Manager
from academy.exchange.local import LocalExchangeFactory
from academy.agent import Agent, loop

from parsl.executors.high_throughput.errors import ManagerLost

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.parsl_configs import combined_config
from pipeline.viral_detection import ViralDetectionAgent
from pipeline.checkv import CheckVAgent
from pipeline.derep_cluster import DereplicationClusteringAgent
from pipeline.blast import BLASTAgent, annotate_blast
from pipeline.tool_selector import ToolSelector
from pipeline import f1_eval


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _fix_parsl_cert_permissions():
    """Pre-create Parsl certificate dirs with mode 0o700 before parsl.load().

    This HPC filesystem's ACL overrides the mode=0o700 that Parsl passes to
    makedirs, producing 0o770 instead. Parsl then refuses to load the
    certificates. Pre-creating the dirs and explicitly chmod-ing them forces
    the correct permissions before Parsl's makedirs (exist_ok=True) runs.
    """
    run_dir = str(config.LOG_DIR / "runinfo")
    os.makedirs(run_dir, exist_ok=True)
    prev = glob(os.path.join(run_dir, "[0-9]*[0-9]"))
    next_num = max(int(os.path.basename(d)) for d in prev) + 1 if prev else 0
    next_rundir = os.path.join(run_dir, f"{next_num:03d}")
    for label in ["viral_htex", "checkv_htex", "derep_htex", "blast_htex"]:
        cert_dir = os.path.join(next_rundir, label, "certificates")
        os.makedirs(cert_dir, exist_ok=True)
        os.chmod(cert_dir, 0o700)


def read_sample_ids(sample_ids_file: str) -> list[str]:
    with open(sample_ids_file) as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_viral_fasta(tool: str, viral_result: str) -> str | None:
    """
    Return the path to a FASTA file from a tool's output, or None if the
    output cannot be resolved to a FASTA for F1 evaluation.

    Most tools return a FASTA path directly. VIBRANT and viralVerify return
    output directories; this function globs for the viral FASTA inside them.
    ViraMiner returns a predictions TXT — F1 evaluation is skipped for it.
    """
    import glob
    if os.path.isfile(viral_result) and viral_result.endswith((".fa", ".fna", ".fasta")):
        return viral_result
    if tool == "VIBRANT" and os.path.isdir(viral_result):
        matches = glob.glob(
            os.path.join(viral_result, "VIBRANT_phages_*", "VIBRANT_phages_*.fna")
        )
        return matches[0] if matches else None
    if tool == "viralVerify" and os.path.isdir(viral_result):
        matches = glob.glob(os.path.join(viral_result, "*_Viral.fasta"))
        return matches[0] if matches else None
    return None


# --------------------------------------------------------------------------- #
# Per-sample pipeline
# --------------------------------------------------------------------------- #

async def process_sample(
    sample_id: str,
    tool: str,
    viral_handle,
    checkv_handle,
    cluster_handle,
    first_sample_id: str,
    ground_truth_viral: set,
) -> tuple:
    """
    Run one sample through viral detection → F1 evaluation → CheckV → dereplication.
    Returns (derep_fasta | None, quality_ratio, f1_score | None).
    derep_fasta is only returned for first_sample_id; others return None.
    """
    unzipped_spades = str(config.SPADES_DIR / sample_id / "assembly.fa")
    profile, model = sample_id.rsplit("_", 1)

    # === Viral Detection ===
    try:
        viral_result = await viral_handle.run_tool(
            tool, unzipped_spades,
            str(config.OUT_GENOMAD / sample_id), str(config.GENOMAD_DB),
            str(config.OUT_VIRSORTER2 / sample_id),
            str(config.OUT_DVF / sample_id), str(config.DVF_DB),
            str(config.WORK_DIR), str(config.WORK_DIR),
            str(config.OUT_MARVEL / sample_id), str(config.MARVEL_DB),
            str(config.OUT_VIRFINDER / sample_id),
            str(config.VIBRANT_DB), str(config.OUT_VIBRANT / sample_id),
            str(config.OUT_VIRALVERIFY / sample_id), str(config.HMM_DB),
            str(config.VIRAMINER_DB), str(config.OUT_VIRAMINER / sample_id),
            str(config.METAPHINDER_DB), config.BLAST_PATH,
            str(config.OUT_METAPHINDER / sample_id),
            str(config.OUT_SEEKER / sample_id),
            str(config.VIRSORTER_DB), str(config.VIRSORTER_SCRIPT),
            str(config.OUT_VIRSORTER / sample_id),
        )
    except ManagerLost as e:
        print(
            f"[{sample_id}] Parsl worker lost during viral detection: {e}"
            f" — skipping sample this round",
            flush=True,
        )
        return None, 0.0, None
    except Exception as e:
        print(
            f"[{sample_id}] Unexpected error during viral detection: {type(e).__name__}: {e}"
            f" — skipping sample this round",
            flush=True,
        )
        return None, 0.0, None

    # === F1 Evaluation (primary tool scoring signal) ===
    f1_score = None
    viral_fasta = _resolve_viral_fasta(tool, viral_result)
    if viral_fasta and os.path.exists(viral_fasta):
        result = f1_eval.evaluate(
            fasta_path=Path(viral_fasta),
            ground_truth_viral=ground_truth_viral,
            tool=tool,
            sample_id=sample_id,
            profile=profile,
            model=model,
        )
        print(result.summary(), flush=True)
        f1_score = result.f1
    else:
        print(
            f"[{sample_id}] Cannot resolve viral FASTA for {tool} — F1 skipped",
            flush=True,
        )

    # === CheckV (quality_ratio kept for future use, not used for tool scoring) ===
    checkv_output_dir = str(config.OUT_CHECKV / sample_id)
    parse_input = os.path.join(checkv_output_dir, "contamination.tsv")
    selection_csv = os.path.join(checkv_output_dir, "selection2_viral.csv")
    subset_spades, quality_ratio = await checkv_handle.run_checkv(
        str(config.CHECKV_PARSER), str(config.PARSE_LENGTH), str(config.WORK_DIR),
        unzipped_spades, viral_result, checkv_output_dir,
        parse_input, selection_csv, str(config.CHECKVDB),
    )
    print(f"[{sample_id}] quality_ratio (CheckV, future use): {quality_ratio:.4f}", flush=True)

    # If CheckV was skipped (no viral sequences), there is nothing to dereplicate.
    if subset_spades is None:
        print(
            f"[{sample_id}] No viral sequences detected — skipping dereplication.",
            flush=True,
        )
        return None, quality_ratio, f1_score

    # === Dereplication ===
    cluster_dir = str(config.OUT_DEREP / sample_id)
    cluster_res_derep = os.path.join(cluster_dir, "clusterRes")
    tmp_dir_derep = os.path.join(cluster_dir, "tmp")
    input_fasta = f"{cluster_res_derep}_all_seqs.fasta"
    cleaned_fasta = os.path.join(cluster_dir, "cleaned_clusterRes_all_seqs.fasta")
    derep_fasta = await cluster_handle.run_dereplicate(
        sample_id, subset_spades, cluster_dir, cluster_res_derep,
        tmp_dir_derep, input_fasta, cleaned_fasta, str(config.OUT_DEREP),
    )

    return (
        derep_fasta if sample_id == first_sample_id else None,
        quality_ratio,
        f1_score,
    )


# --------------------------------------------------------------------------- #
# Coordinator Agent
# --------------------------------------------------------------------------- #

class CoordinatorAgent(Agent):
    def __init__(
        self,
        viral_handle,
        checkv_handle,
        cluster_handle,
        blast_handle,
        shutdown_event: asyncio.Event,
    ):
        super().__init__()
        self.viral_handle   = viral_handle
        self.checkv_handle  = checkv_handle
        self.cluster_handle = cluster_handle
        self.blast_handle   = blast_handle
        self.selector       = ToolSelector(alpha=0.6)
        self.current_tool   = "VirSorter"
        self.f1_history            = []
        self.quality_ratios_history = []  # kept for future use
        self.match_ratios_history   = []  # kept for future use
        self.shutdown = shutdown_event

    @loop
    async def continuous_pipeline(self, shutdown: asyncio.Event) -> None:
        round_count = 0
        while not shutdown.is_set():
            print(
                f"\n=== [Coordinator] Round {round_count + 1} | Tool: {self.current_tool} ===",
                flush=True,
            )

            sample_ids = read_sample_ids(str(config.XFILE_DIR / config.XFILE))
            if not sample_ids:
                raise RuntimeError(f"No sample IDs found in {config.XFILE_DIR / config.XFILE}")
            first_sample_id = sample_ids[0]

            # Pre-load ground truth once per (profile, model) — CSV is 1.1 M rows
            gt_cache: dict[tuple, set] = {}
            for sid in sample_ids:
                profile, model = sid.rsplit("_", 1)
                key = (profile, model)
                if key not in gt_cache:
                    gt_cache[key] = f1_eval.load_viral_contig_ids(
                        config.CONTIG_IDENTITIES, profile, model
                    )

            # === Per-sample: viral detection + F1 + CheckV + dereplication ===
            per_sample_tasks = [
                asyncio.create_task(
                    process_sample(
                        sid, self.current_tool,
                        self.viral_handle, self.checkv_handle, self.cluster_handle,
                        first_sample_id,
                        gt_cache[tuple(sid.rsplit("_", 1))],
                    )
                )
                for sid in sample_ids
            ]
            raw_results = await asyncio.gather(*per_sample_tasks, return_exceptions=True)

            # Separate successful results from any unhandled exceptions (defence in depth)
            results = []
            for sid, res in zip(sample_ids, raw_results):
                if isinstance(res, BaseException):
                    print(
                        f"[Coordinator] [{sid}] unhandled exception escaped process_sample:"
                        f" {type(res).__name__}: {res}",
                        flush=True,
                    )
                else:
                    results.append(res)

            if not results:
                print(
                    "[Coordinator] All samples failed this round — retrying next round.",
                    flush=True,
                )
                await asyncio.sleep(10)
                continue

            # Collect quality ratios (kept for future use)
            quality_ratios = [qr for _, qr, _ in results if qr is not None]
            avg_quality_ratio = sum(quality_ratios) / len(quality_ratios) if quality_ratios else 0.0
            self.quality_ratios_history.append(avg_quality_ratio)
            print(
                f"[Coordinator] Avg quality_ratio (CheckV, future use): {avg_quality_ratio:.4f}",
                flush=True,
            )

            # Collect F1 scores
            f1_scores = [f1 for _, _, f1 in results if f1 is not None]
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            self.f1_history.append(avg_f1)
            print(f"[Coordinator] Avg F1: {avg_f1:.4f}", flush=True)

            # === Cluster ===
            derep_fasta = next((df for df, _, _ in results if df is not None), None)
            if derep_fasta is None:
                print("[Coordinator] Skipping round — no dereplicated FASTA.", flush=True)
                await asyncio.sleep(10)
                continue

            out_cluster = str(config.OUT_CLUSTER)
            work_dir    = str(config.WORK_DIR)
            cluster_res_cluster = os.path.join(out_cluster, "clusterRes")
            tmp_dir_cluster     = os.path.join(out_cluster, "tmp")
            rep_seq_src = os.path.join(out_cluster, "clusterRes_rep_seq.fasta")
            rep_seq_dst = os.path.join(work_dir, "query")

            query_dir, cluster_file = await self.cluster_handle.run_cluster(
                sample_ids, str(config.OUT_DEREP), derep_fasta, out_cluster,
                cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst,
            )

            # === BLAST (match_ratio kept for future use) ===
            db_dir         = str(config.DB_DIR)
            db_list_path   = os.path.join(db_dir, "db-list")
            results_dir    = os.path.join(work_dir, "results_testing", "05B_launchblast")
            blast_results_dir  = os.path.join(work_dir, "results_testing", "05C_blast")
            merge_results_dir  = os.path.join(work_dir, "results_testing", "05D_mergeblast")

            hits_file, match_ratio = await self.blast_handle.run_full_blast(
                work_dir, config.FA_SPLIT_FILE_SIZE, results_dir,
                query_dir, cluster_file, db_dir,
                blast_results_dir, config.BLAST_TYPE, config.EVAL,
                config.OUT_FMT, config.MAX_TARGET_SEQS,
                merge_results_dir, config.MAX_DB_SIZE, db_list_path,
            )
            self.match_ratios_history.append(match_ratio)
            print(
                f"[Coordinator] match_ratio (BLAST, future use): {match_ratio:.4f}",
                flush=True,
            )

            # === Tool scoring — F1 is the primary signal ===
            self.selector.update_tool_score(self.current_tool, avg_f1)
            # Exclude ViraMiner: its predictions TXT has no FASTA extraction yet,
            # so it would always score 0.0 and bias selection results.
            operational_tools = [t for t in ToolSelector.TOOLS if t != "ViraMiner"]
            self.current_tool = self.selector.choose_tool(operational_tools)
            print(f"[Coordinator] Next tool: {self.current_tool}", flush=True)

            # === Annotation ===
            script_path = os.path.join(work_dir, "solution1_manual.py")
            if not os.path.exists(script_path):
                print(
                    f"[Coordinator] annotation script not found at {script_path} — skipping",
                    flush=True,
                )
            else:
                final = annotate_blast(
                    hits_file, str(config.ANNOTATIONS), str(config.OUTPUT),
                    script_path, config.PCTID, config.LENGTH,
                )
                print("[Coordinator]", final, flush=True)

            round_count += 1
            if round_count >= 10:
                print("[Coordinator] 10 rounds complete. Shutting down.", flush=True)
                self.shutdown.set()
                final_avg_f1      = sum(self.f1_history) / len(self.f1_history) if self.f1_history else 0.0
                final_avg_quality = sum(self.quality_ratios_history) / len(self.quality_ratios_history) if self.quality_ratios_history else 0.0
                final_avg_match   = sum(self.match_ratios_history) / len(self.match_ratios_history) if self.match_ratios_history else 0.0
                print(f"\n[Coordinator] FINAL avg F1 over 10 rounds:           {final_avg_f1:.4f}", flush=True)
                print(f"[Coordinator] FINAL avg quality_ratio (future use):   {final_avg_quality:.4f}", flush=True)
                print(f"[Coordinator] FINAL avg match_ratio   (future use):   {final_avg_match:.4f}", flush=True)
                print(f"[Coordinator] Best tool: {self.selector.best_tool}", flush=True)
                break

            await asyncio.sleep(5)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

async def main():
    start_time = datetime.now()
    print(f"[Main] Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Load a single combined Parsl config with all four named executors before
    # any agents are created. Individual agent __init__ methods must NOT call
    # parsl.clear()/parsl.load() — doing so would destroy this config.
    _fix_parsl_cert_permissions()
    parsl.load(combined_config)

    shutdown_event = asyncio.Event()

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        viral_handle   = await manager.launch(ViralDetectionAgent())
        checkv_handle  = await manager.launch(CheckVAgent())
        cluster_handle = await manager.launch(DereplicationClusteringAgent())
        blast_handle   = await manager.launch(BLASTAgent())

        coord = CoordinatorAgent(viral_handle, checkv_handle, cluster_handle, blast_handle, shutdown_event)
        await manager.launch(coord)
        await shutdown_event.wait()
        print("[Main] Shutdown complete.", flush=True)

    end_time = datetime.now()
    print(f"[Main] End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[Main] Total: {end_time - start_time}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
