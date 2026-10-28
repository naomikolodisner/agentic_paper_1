#!/usr/bin/env nextflow

// Classify paired FMT reads taxonomically against the HumGut kraken2 database
 process kraken2 {
    label "process_high"
    label "publish_final"
    container "${params.container__kraken2}"

    input:
        tuple val(sampleid), path(r1), path(r2)
        path(kraken2_db)

    output:
        tuple val(sampleid), path("${sampleid}.report"), path("${sampleid}.kraken"), emit: classification

    shell:
    '''
    kraken2 \
    --db !{kraken2_db} \
    --threads !{task.cpus} \
    --paired \
    --use-names \
    --output !{sampleid}.kraken \
    --report !{sampleid}.report \
    !{r1} !{r2}
    '''
}
