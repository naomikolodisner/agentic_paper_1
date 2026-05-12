#!/bin/bash
#SBATCH --job-name=viral_benchmark
#SBATCH --output=/xdisk/gwatts/kolodisner/agentic_paper_1/logs/slurm/viral_benchmark_%j.log
#SBATCH --error=/xdisk/gwatts/kolodisner/agentic_paper_1/logs/slurm/viral_benchmark_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --partition=standard
#SBATCH --account=gwatts
#SBATCH --cpus-per-task=4

CONDA="/groups/gwatts/miniconda3"
source $CONDA/etc/profile.d/conda.sh
conda activate academy_py311

cd /xdisk/gwatts/kolodisner/agentic_paper_1
python -m pipeline.coordinator
