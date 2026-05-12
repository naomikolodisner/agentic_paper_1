import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from academy.manager import Manager
from academy.exchange.local import LocalExchangeFactory
from academy.agent import Agent, loop

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.viral_detection import ViralDetectionAgent
from pipeline.checkv import CheckVAgent
from pipeline.derep_cluster import DereplicationClusteringAgent
from pipeline.blast import BLASTAgent, annotate_blast
from pipeline.tool_selector import ToolSelector
from pipeline import f1_eval


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def read_sample_ids(sample_ids_file: str) -> list[str]:
    with open(sample_ids_file) as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_viral_fasta(tool: str, viral_result: str) -> str | None:
    """
    Return the path to a FASTA file from a tool's output, or None if the
    output format is not directly parseable as FASTA.

    Most tools return a FASTA path. VIBRANT and viralVerify return output
    directories; ViraMiner returns a predictions TXT. F1 evaluation is skipped
    for those until per-tool extraction is implemented.
    """
    if os.path.isfile(viral_result) and viral_result.endswith((".fa", ".fna", ".fasta")):
        return viral_result
    # TODO: add FASTA extraction for VIBRANT (directory), viralVerify (directory),
    #       and ViraMiner (predictions TXT)
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

    # === Dereplication ===
    cluster_dir = str(config.OUT_DEREP / sample_id)
    cluster_res_derep = os.path.join(cluster_dir, "clusterRes")
    tmp_dir_derep = os.path.join(cluster_dir, "tmp")
    input_fasta = f"{cluster_res_derep}_all_seqs.fasta"
    cleaned_fasta = os.path.join(cluster_dir, "cleaned_clusterRes_all_seqs.fasta")
    derep_fasta = await (await cluster_handle.run_dereplicate(
        sample_id, subset_spades, cluster_dir, cluster_res_derep,
        tmp_dir_derep, input_fasta, cleaned_fasta, str(config.OUT_DEREP),
    ))

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
            results = await asyncio.gather(*per_sample_tasks)

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

            query_dir, cluster_file = await (await self.cluster_handle.run_cluster(
                sample_ids, str(config.OUT_DEREP), derep_fasta, out_cluster,
                cluster_res_cluster, tmp_dir_cluster, rep_seq_src, rep_seq_dst,
            ))

            # === BLAST (match_ratio kept for future use) ===
            db_dir         = str(config.DB_DIR)
            db_list_path   = os.path.join(db_dir, "db-list")
            results_dir    = os.path.join(work_dir, "results_testing", "05B_launchblast")
            blast_results_dir  = os.path.join(work_dir, "results_testing", "05C_blast")
            merge_results_dir  = os.path.join(work_dir, "results_testing", "05D_mergeblast")

            hits_file, match_ratio = await (await self.blast_handle.run_full_blast(
                work_dir, config.FA_SPLIT_FILE_SIZE, results_dir,
                query_dir, cluster_file, db_dir,
                blast_results_dir, config.BLAST_TYPE, config.EVAL,
                config.OUT_FMT, config.MAX_TARGET_SEQS,
                merge_results_dir, config.MAX_DB_SIZE, db_list_path,
            ))
            self.match_ratios_history.append(match_ratio)
            print(
                f"[Coordinator] match_ratio (BLAST, future use): {match_ratio:.4f}",
                flush=True,
            )

            # === Tool scoring — F1 is the primary signal ===
            self.selector.update_tool_score(self.current_tool, avg_f1)
            self.current_tool = self.selector.choose_tool(
                ["VirSorter2", "DeepVirFinder", "GeNomad", "MARVEL"]
            )
            print(f"[Coordinator] Next tool: {self.current_tool}", flush=True)

            # === Annotation ===
            script_path = os.path.join(work_dir, "solution1_manual.py")
            final = annotate_blast(
                hits_file, str(config.ANNOTATIONS), str(config.OUTPUT),
                script_path, config.PCTID, config.LENGTH,
            )
            print("[Coordinator]", final, flush=True)

            round_count += 1
            if round_count >= 10:
                print("[Coordinator] 10 rounds complete. Shutting down.", flush=True)
                self.shutdown.set()
                final_avg_f1      = sum(self.f1_history) / len(self.f1_history)
                final_avg_quality = sum(self.quality_ratios_history) / len(self.quality_ratios_history)
                final_avg_match   = sum(self.match_ratios_history) / len(self.match_ratios_history)
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

    shutdown_event = asyncio.Event()

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        viral_handle   = await manager.launch(ViralDetectionAgent())
        checkv_handle  = await manager.launch(CheckVAgent())
        cluster_handle = await manager.launch(DereplicationClusteringAgent())
        blast_handle   = await manager.launch(BLASTAgent())

        await manager.launch(
            CoordinatorAgent,
            args=(viral_handle, checkv_handle, cluster_handle, blast_handle, shutdown_event),
        )
        await shutdown_event.wait()
        print("[Main] Shutdown complete.", flush=True)

    end_time = datetime.now()
    print(f"[Main] End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[Main] Total: {end_time - start_time}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
