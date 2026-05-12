from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.usage_tracking.levels import LEVEL_1

_LOG_DIR = "/xdisk/gwatts/kolodisner/agentic_paper_1/logs/runinfo"

viral_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="Parsl_htex",
            worker_debug=False,
            cores_per_worker=16.0,
            max_workers_per_node=4,
            provider=SlurmProvider(
                partition='standard',
                account='gwatts',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=94,
                nodes_per_block=1,
                scheduler_options='',
                exclusive=True,
                walltime='12:00:00',
                launcher=SrunLauncher(),
                worker_init='',
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

checkv_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="Parsl_htex",
            worker_debug=False,
            cores_per_worker=16.0,
            max_workers_per_node=3,
            provider=SlurmProvider(
                partition='standard',
                account='gwatts',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=48,
                nodes_per_block=1,
                scheduler_options='#SBATCH --time=12:00:00',
                exclusive=True,
                cmd_timeout=60 * 60 * 12,
                walltime='4:00:00',
                launcher=SrunLauncher(overrides="--time=4:00:00"),
                worker_init='',
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

derep_cluster_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="Parsl_htex",
            worker_debug=False,
            cores_per_worker=1.0,
            max_workers_per_node=94,
            provider=SlurmProvider(
                partition='standard',
                account='gwatts',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=94,
                nodes_per_block=1,
                scheduler_options='',
                cmd_timeout=60,
                walltime='1:00:00',
                launcher=SrunLauncher(),
                worker_init='',
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)

blast_config = Config(
    run_dir=_LOG_DIR,
    executors=[
        HighThroughputExecutor(
            label="Parsl_htex",
            worker_debug=False,
            cores_per_worker=1.0,
            max_workers_per_node=94,
            provider=SlurmProvider(
                partition='standard',
                account='gwatts',
                init_blocks=1,
                mem_per_node=80,
                cores_per_node=94,
                nodes_per_block=1,
                scheduler_options='',
                cmd_timeout=60,
                walltime='1:00:00',
                launcher=SrunLauncher(),
                worker_init='',
            ),
        )
    ],
    usage_tracking=LEVEL_1,
)
