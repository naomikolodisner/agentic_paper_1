#!/usr/bin/env bash
#SBATCH --job-name=create_samples
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1

# DEBUG INFO
echo "Job started at $(date)"
echo "Running on node $(hostname)"
echo "User: $USER"
echo "PWD: $PWD"
echo "SLURM JobID: $SLURM_JOB_ID"
echo "SLURM_MEM_PER_CPU=$SLURM_MEM_PER_CPU"
echo "PATH=$PATH"
echo "Conda base info:"
GEN_TITRATION="/xdisk/gwatts/kolodisner/agentic_paper_1/scripts/gen_titration_sample.py"

# Activate conda first
source /groups/gwatts/miniconda3/etc/profile.d/conda.sh
conda activate test_env
echo "Conda activated"

# Now check commands
echo "Python: $(which python)"
echo "ART: $(which art_illumina)"
if [ ! -x "$GEN_TITRATION" ]; then
    echo "ERROR: gen_titration_sample.py not executable or missing at $GEN_TITRATION"
fi

# Directories
SAMPLES_DIR="/xdisk/gwatts/kolodisner/agentic_paper_1/dataset_creation/samples/single"
BACKGROUND_DIR="/xdisk/gwatts/kolodisner/agentic_paper_1/dataset_creation/no_virus_contigs"

mkdir -p log tmp logs

# Loop over samples
for sample_dir in "${SAMPLES_DIR}"/*; do
    [ -d "$sample_dir" ] || continue

    sample_name=$(basename "${sample_dir}")
    log_file="log/${sample_name}.log"

    echo "Processing ${sample_name}" | tee -a "$log_file"

    # TMPDIR for this sample
    rand_dir=$(tr -dc 'a-zA-Z0-9' </dev/urandom | fold -w 8 | head -n 1)
    TMPDIR="${PWD}/tmp_${sample_name}_${rand_dir}"
    mkdir -p "${TMPDIR}"
    echo "TMPDIR for sample: ${TMPDIR}" | tee -a "$log_file"

    # Check input FASTA
    contigs_fasta="${sample_dir}/contigs.fasta"
    if [ ! -f "${contigs_fasta}" ]; then
        echo "ERROR: Missing contigs file: ${contigs_fasta}" | tee -a "$log_file"
        continue
    else
        echo "Found contigs: ${contigs_fasta}" | tee -a "$log_file"
        ls -lh "${contigs_fasta}" | tee -a "$log_file"
    fi

    coverages=(0.1 0.5 1 10)

    for cov in "${coverages[@]}"; do
        echo "  Coverage ${cov}x" | tee -a "$log_file"

        sample_out_dir="${sample_dir}/${cov}x"
        mkdir -p "${sample_out_dir}"

        # Simulate sample reads
        echo "  Running ART on sample..." | tee -a "$log_file"
        echo "  Input FASTA: ${contigs_fasta}" | tee -a "$log_file"
        /usr/bin/time -v art_illumina -ss HS25 -na \
            -i "${contigs_fasta}" \
            -f "${cov}" \
            -l 100 -m 200 -s 10 \
            -o "${TMPDIR}/sample" 2>&1 | tee -a "$log_file"

        # Check if ART produced output
        if [ ! -f "${TMPDIR}/sample1.fq" ] || [ ! -f "${TMPDIR}/sample2.fq" ]; then
            echo "ERROR: ART did not produce output FASTQ files!" | tee -a "$log_file"
            ls -lh "${TMPDIR}" | tee -a "$log_file"
            continue
        fi

        mv "${TMPDIR}/sample1.fq" "${TMPDIR}/sample.1.fq"
        mv "${TMPDIR}/sample2.fq" "${TMPDIR}/sample.2.fq"

        # Pick random background
        background_fasta=$(ls ${BACKGROUND_DIR}/*.fasta | shuf -n 1)
        if [ ! -f "${background_fasta}" ]; then
            echo "ERROR: Missing background file" | tee -a "$log_file"
            continue
        fi
        echo "Using background FASTA: ${background_fasta}" | tee -a "$log_file"

        # Simulate background reads
        echo "  Running ART on background..." | tee -a "$log_file"
        /usr/bin/time -v art_illumina \
            -i "${background_fasta}" \
            -f 10 \
            -l 100 -m 200 -s 10 \
            -o "${TMPDIR}/background" 2>&1 | tee -a "$log_file"

        if [ ! -f "${TMPDIR}/background1.fq" ] || [ ! -f "${TMPDIR}/background2.fq" ]; then
            echo "ERROR: ART background did not produce FASTQ files!" | tee -a "$log_file"
            ls -lh "${TMPDIR}" | tee -a "$log_file"
            continue
        fi

        # Combine sample + background
        echo "  Running gen_titration_sample.py..." | tee -a "$log_file"
        python "$GEN_TITRATION" \
            -R1 "${TMPDIR}/sample.1.fq" \
            -R2 "${TMPDIR}/sample.2.fq" \
            -B1 "${TMPDIR}/background1.fq" \
            -B2 "${TMPDIR}/background2.fq" \
            --depth 3000000000 \
            -o "${sample_out_dir}/sample" 2>&1 | tee -a "$log_file" || {
            echo "ERROR: gen_titration_sample.py failed for ${sample_name} coverage ${cov}" | tee -a "$log_file"
            continue
            }

        echo "  Completed coverage ${cov}x for ${sample_name}" | tee -a "$log_file"
        ls -lh "${sample_out_dir}" | tee -a "$log_file"

    done

    # TMPDIR is kept for debugging
    echo "TMPDIR kept at ${TMPDIR} for inspection" | tee -a "$log_file"
done

echo "All samples complete at $(date)"
