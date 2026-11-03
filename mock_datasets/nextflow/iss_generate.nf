#!/usr/bin/env nextflow

// Simulate paired metagenome reads with InSilicoSeq from draft genomes (one file = one genome) and an abundance file
process create_mock_community_reads_draft_abundance_file {
    label "process_high"
    label "publish_intermediate"
    container "${params.container__insilicoseq}"

    input:
        tuple val(sampleid), val(model), val(n_reads), val(seed), path(abundance_tsv), path(genomes)

    output:
        tuple val(sampleid), val(model), path("${sampleid}_R1.fastq.gz"), path("${sampleid}_R2.fastq.gz"), env(read_len), emit: reads

    shell:
    '''
    awk -F'\\t' -v OFS='\\t' '{ n = split($1, p, "/"); $1 = p[n]; print }' !{abundance_tsv} > abundance.tsv

    iss generate \
    --draft $(cut -f1 abundance.tsv) \
    --abundance_file abundance.tsv \
    --model !{model} \
    --n_reads !{n_reads} \
    --seed !{seed} \
    --cpus !{task.cpus} \
    --compress \
    --output !{sampleid}

    read_len=$( (zcat !{sampleid}_R1.fastq.gz || true) | awk 'NR == 2 { print length($0); exit }')
    '''
}
