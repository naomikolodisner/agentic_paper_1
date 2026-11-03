#!/usr/bin/env nextflow

// Classify paired reads taxonomically against a kraken2 database.
// The workflow shapes the reads into one tuple per sample, e.g. from a sample list
// with reads at <reads_dir>/<sample_id>/r1.fastq.gz and r2.fastq.gz:
//   Channel.fromPath(params.sample_list).splitText().map { it.trim() }
//       .map { id -> tuple(id, file("${params.reads_dir}/${id}/r1.fastq.gz"),
//                              file("${params.reads_dir}/${id}/r2.fastq.gz")) }
process kraken2 {
    label "process_high"
    label "publish_intermediate"
    container "${params.container__kraken2}"

    input:
        tuple val(sample_id), path(r1), path(r2)
        path(kraken2_db)

    output:
        tuple val(sample_id), path("${sample_id}.report"), emit: report
        tuple val(sample_id), path("${sample_id}.kraken"), emit: kraken

    shell:
    '''
    kraken2 \
    --db !{kraken2_db} \
    --threads !{task.cpus} \
    --paired \
    --use-names \
    --output !{sample_id}.kraken \
    --report !{sample_id}.report \
    !{r1} !{r2}
    '''
}
// view, map, collect --> functions that allow us to change shape out inputs/outputs
// emit vs publish
// inputs made into a channel, importing data from path into workflow
// next flow does everything isde the workflow unless you publish, keeps in work directory
// if you give nextflow an output directory, by default simlink will be the publish method
// outdirectory will contain symlinks to work directory (in scratch) --> dont always want this
//HW: keep making processes --> take rest of tools and create processes
//create a pull request for the nextflow process library and submit process to it
// if there is a process there, look at if i think its correct, and if i dont think its correct, make changes
// make our own branch and then make a pull request to that branch, mery will revise it and merge it to main
// start with just tools, and then we will talk about parsers, will talk about those next week,
//make a branch, commit to the branch, make a pull request to main
//can fork (helps Mery!) or you can just clone it and make a branch
// have to fork if you are not a member of the repo
// if you want to have actual final in outdir change mode to publish as mode = copy, better to save all outputs to workdir at the end though because sctach is part of HPC and rs1 is a separate hard drive
