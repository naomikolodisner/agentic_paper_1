#!/usr/bin/env bash
#SBATCH --account=gwatts
#SBATCH --partition=standard
#SBATCH --job-name=create_samples_array
#SBATCH --output=/xdisk/gwatts/kolodisner/agentic_paper_1/logs/slurm/slurm-%A_%a.out
#SBATCH --error=/xdisk/gwatts/kolodisner/agentic_paper_1/logs/slurm/slurm-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-14%3

echo "Job started at $(date)"
echo "Array JobID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node $(hostname)"

# Activate conda
#source /groups/gwatts/miniconda3/etc/profile.d/conda.sh
#conda activate test
#echo "Conda activated"

GEN_TITRATION="/xdisk/gwatts/kolodisner/agentic_paper_1/scripts/gen_titration_sample.py"
SAMPLES_DIR="/xdisk/gwatts/kolodisner/agentic_paper_1/data/spike_in_samples/single"
BACKGROUND_DIR="/xdisk/gwatts/kolodisner/agentic_paper_1/data/no_virus_contigs"
ART_BIN="/home/u3/kolodisner/.conda/envs/test/bin/art_illumina"
PYTHON_BIN="$HOME/.conda/envs/test/bin/python"

mkdir -p /xdisk/gwatts/kolodisner/agentic_paper_1/logs/slurm /xdisk/gwatts/kolodisner/agentic_paper_1/logs/sample

# Get list of sample directories
mapfile -t SAMPLE_DIRS < <(find "$SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# Check index
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SAMPLE_DIRS[@]}" ]; then
    echo "Task ID exceeds number of samples. Exiting."
    exit 0
fi

sample_dir="${SAMPLE_DIRS[$SLURM_ARRAY_TASK_ID]}"
sample_name=$(basename "$sample_dir")
log_file="/xdisk/gwatts/kolodisner/agentic_paper_1/logs/sample/${sample_name}.log"
: > "$log_file"   

echo "Processing ${sample_name}" | tee "$log_file"

# TMPDIR setup
if [ -z "$SLURM_TMPDIR" ]; then
    TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    TMPDIR="$SLURM_TMPDIR"
fi
mkdir -p "$TMPDIR"

# Limit hidden threading
export OMP_NUM_THREADS=1
echo "TMPDIR: $TMPDIR" | tee -a "$log_file"

# Cleanup on exit
trap "rm -rf $TMPDIR" EXIT
echo "TMPDIR exists? $(ls -ld $TMPDIR)" | tee -a "$log_file"


# Input FASTA
contigs_fasta="${sample_dir}/contigs.fasta"
if [ ! -f "$contigs_fasta" ]; then
    echo "ERROR: Missing contigs file" | tee -a "$log_file"
    exit 1
fi

coverages=(0.1 0.5 1 10)

for cov in "${coverages[@]}"; do
    echo "Coverage ${cov}x" | tee -a "$log_file"

    sample_out_dir="${sample_dir}/${cov}x"
    mkdir -p "$sample_out_dir"

    # ART - sample
    /usr/bin/time -v "$ART_BIN" -ss HS25 -na \
        -i "$contigs_fasta" \
        -f "$cov" \
        -l 100 -m 200 -s 10 \
        -o "$TMPDIR/sample" 2>&1 | tee -a "$log_file"

    if [ ! -f "$TMPDIR/sample1.fq" ]; then
        echo "ERROR: ART sample failed" | tee -a "$log_file"
        continue
    fi

    mv "$TMPDIR/sample1.fq" "$TMPDIR/sample.1.fq"
    mv "$TMPDIR/sample2.fq" "$TMPDIR/sample.2.fq"

    # Random background
    background_fasta=$(ls ${BACKGROUND_DIR}/*/contigs.fasta | shuf -n 1)

    # ART - background
    /usr/bin/time -v "$ART_BIN" \
        -i "$background_fasta" \
        -f 10 \
        -l 100 -m 200 -s 10 \
        -o "$TMPDIR/background" 2>&1 | tee -a "$log_file"

    if [ ! -f "$TMPDIR/background1.fq" ]; then
        echo "ERROR: ART background failed" | tee -a "$log_file"
        continue
    fi

    # Combine
    /usr/bin/time -v "$PYTHON_BIN" "$GEN_TITRATION" \
        -R1 "$TMPDIR/sample.1.fq" \
        -R2 "$TMPDIR/sample.2.fq" \
        -B1 "$TMPDIR/background1.fq" \
        -B2 "$TMPDIR/background2.fq" \
        --depth 3000000000 \
        -o "${sample_out_dir}/sample" \
        2>&1 | tee -a "$log_file"

    echo "Completed ${sample_name} ${cov}x" | tee -a "$log_file"
    # Clean temp files between coverages
    rm -f $TMPDIR/*.fq
done

echo "Done with ${sample_name}"
