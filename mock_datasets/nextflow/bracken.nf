#!/usr/bin/env nextflow

// Build Bracken's kmer-distribution file for a kraken2 database (needs library/, taxonomy/ and seqid2taxid.map, as made by kraken2-build)
process bracken_build {
    label "process_high"
    container "${params.container__bracken}"
    publishDir "${params.bracken_build_outdir}", mode: 'copy'

    input:
        path(kraken2_db)
        val(read_len)

    output:
        path("${kraken2_db}/database${read_len}mers.kmer_distrib"), emit: kmer_distrib

    shell:
    '''
    bracken-build \
    -d !{kraken2_db} \
    -t !{task.cpus} \
    -k !{params.bracken_build_kmer_len} \
    -l !{read_len}
    '''
}

// Re-estimate abundances from a kraken2 report with Bracken, moving reads that
// kraken2 left at an ancestor rank back down to species level ("S").
// In the HumGut DB every genome has its own S-rank node, so S means per-genome.
// read_len must match the one the kmer_distrib was built for (bracken_build).
process bracken {
    label "process_low"
    label "publish_intermediate"
    container "${params.container__bracken}"

    input:
        tuple val(sample_id), path(kraken_report)
        path(kmer_distrib)
        val(read_len)

    output:
        tuple val(sample_id), path("${sample_id}.bracken"), emit: bracken
        tuple val(sample_id), path("${sample_id}_bracken.report"), emit: report

    shell:
    '''
    # bracken looks for <db>/database<read_len>mers.kmer_distrib, so -d . finds the staged file
    bracken \
    -d . \
    -i !{kraken_report} \
    -o !{sample_id}.bracken \
    -w !{sample_id}_bracken.report \
    -r !{read_len} \
    -l S \
    -t 10
    '''
}
