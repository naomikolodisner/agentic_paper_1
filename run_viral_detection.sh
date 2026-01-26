#!/bin/bash
#SBATCH --job-name=viral_benchmark
#SBATCH --output=viral_benchmark_%j.log
#SBATCH --error=viral_benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G


#load environment
CONDA="/groups/gwatts/miniconda3"
source $CONDA/etc/profile.d/conda.sh
conda activate academy_py311

python /xdisk/gwatts/kolodisner/agentic_paper_1/agentic_viral_benchmark.py
