#!/usr/bin/env nextflow

// Classify paired reads taxonomically against a kraken2 database.
// Reads are expected at <reads_dir>/<sample_id>/r1.fastq.gz and r2.fastq.gz
 process kraken2 {
    label "process_high"
    label "publish_final"
    conda "${params.conda__kraken2}"

    input:
        val(sample_id)
        path(reads_dir)
        path(kraken2_db)

    output:
        path("${sample_id}.report"), emit: report
        path("${sample_id}.kraken"), emit: kraken

    shell:
    '''
    kraken2 \
    --db !{kraken2_db} \
    --threads !{task.cpus} \
    --paired \
    --use-names \
    --output !{sample_id}.kraken \
    --report !{sample_id}.report \
    !{reads_dir}/!{sample_id}/r1.fastq.gz !{reads_dir}/!{sample_id}/r2.fastq.gz
    '''
}
