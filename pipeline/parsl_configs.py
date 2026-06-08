import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.usage_tracking.levels import LEVEL_1

_LOG_DIR = str(config.LOG_DIR / "runinfo")
_PROJECT_ROOT = str(config.PROJECT_ROOT)
_CONDA_ROOT = "/usr/local/apps/miniconda20240526"

# Executed on each compute node before workers start.
# 1. Activates the main conda env so the right Python/packages are available.
# 2. Exports PYTHONPATH so worker processes can import the `pipeline` package.
_WORKER_INIT = (
    f"source {_CONDA_ROOT}/etc/profile.d/conda.sh && "
    f"conda activate academy_py311 && "
    f"export PYTHONPATH={_PROJECT_ROOT}:$PYTHONPATH"
)

viral_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="viral_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=16.0,
            # Reduced from 4 → 2 workers per node so each worker gets ~80 GB instead
            # of ~20 GB.  Heavy tools (geNomad, VirSorter2) were OOM-killing the manager.
            max_workers_per_node=2,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                # Increased from 80 → 160 GB (Puma standard nodes have 256 GB).
                mem_per_node=160,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                exclusive=True,
                walltime='12:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

checkv_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="checkv_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=16.0,
            max_workers_per_node=2,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                exclusive=True,
                walltime='4:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

derep_cluster_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="derep_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=1.0,
            max_workers_per_node=32,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                walltime='1:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

blast_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="blast_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=1.0,
            max_workers_per_node=32,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                walltime='1:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

# Combined config for loading once in main() with all four named executors.
# Each @python_app is decorated with executors=['<label>'] to route to the
# correct executor without each agent independently calling parsl.load().
combined_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="viral_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=16.0,
            max_workers_per_node=2,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=160,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                exclusive=True,
                walltime='12:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        ),
        HighThroughputExecutor(
            label="checkv_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=16.0,
            max_workers_per_node=2,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                exclusive=True,
                walltime='4:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        ),
        HighThroughputExecutor(
            label="derep_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=1.0,
            max_workers_per_node=32,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                walltime='1:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        ),
        HighThroughputExecutor(
            label="blast_htex",
            worker_debug=False,
            encrypted=False,
            cores_per_worker=1.0,
            max_workers_per_node=32,
            provider=SlurmProvider(
                partition='compute',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=32,
                nodes_per_block=1,
                scheduler_options='',
                walltime='1:00:00',
                launcher=SrunLauncher(),
                worker_init=_WORKER_INIT,
            ),
        ),
    ],
    usage_tracking=LEVEL_1,
)
