#!/usr/bin/env nextflow

// Simulate paired metagenome reads with InSilicoSeq from draft genomes and an
// abundance profile (genome path<TAB>abundance, no header).
// ISS matches the profile's first column against the literal --draft strings, so
// the profile's paths are rewritten to the staged file names and --draft is built
// from that same column. Each --draft file is modeled as one genome.
// genomes is every file listed in the profile, e.g.
//   profile.splitCsv(sep: "\t").map { file(it[0]) }.collect()
// Also emits the measured read length, which ART needs to match the background.
process iss_generate {
    label "process_high"
    label "publish_intermediate"
    container "${params.container__insilicoseq}"

    input:
        tuple val(sample_id), val(model), val(n_reads), val(seed), path(abundance_tsv), path(genomes)

    output:
        tuple val(sample_id), val(model), path("${sample_id}_R1.fastq.gz"), path("${sample_id}_R2.fastq.gz"), env(read_len), emit: reads

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
    --output !{sample_id}

    read_len=$( (zcat !{sample_id}_R1.fastq.gz || true) | awk 'NR == 2 { print length($0); exit }')
    '''
}
