#!/usr/bin/env nextflow

// Simulate paired-end Illumina reads from every sequence in a FASTA at a fold coverage with ART
process art_illumina {
    label "process_low"
    label "publish_intermediate"
    container "${params.container__art}"

    input:
        tuple val(sampleid), path(fasta), val(fold_coverage)

    output:
        tuple val(sampleid), path("${sampleid}_1.fq"), path("${sampleid}_2.fq"), emit: reads

    shell:
    '''
    art_illumina \
    -ss !{params.art_illumina_seq_sys} \
    -p \
    -na \
    -i !{fasta} \
    -f !{fold_coverage} \
    -l !{params.art_illumina_len} \
    -m !{params.art_illumina_mflen} \
    -s !{params.art_illumina_sdev} \
    -o !{sampleid}_
    '''
}
