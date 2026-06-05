#!/bin/bash
#SBATCH --job-name=viral_benchmark
#SBATCH --output=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/viral_benchmark_%j.log
#SBATCH --error=/rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1/logs/slurm/viral_benchmark_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

CONDA="/usr/local/apps/miniconda20240526"
source $CONDA/etc/profile.d/conda.sh
conda activate academy_py311

cd /rs1/researchers/b/blhurwit/users/nkolodi/agentic_paper_1
python -m pipeline.coordinator
