#!/usr/bin/env nextflow

// Build Bracken's kmer-distribution file for an existing kraken2 database.
// bracken-build needs <db>/library/*.fna, <db>/taxonomy/{nodes,names}.dmp and
// <db>/seqid2taxid.map, and writes its output into <db>. The process builds that
// layout in the work dir (symlinking the *.k2d files), so the input DB is never modified.
// Genome headers must follow kraken2's ">kraken:taxid|<taxid>|<seqid>" convention.
// seqid2taxid.map is keyed on that WHOLE first token, since that is the sequence ID
// kraken2 classify reports; keying on the bare <seqid> makes every lookup fail.
process bracken_build {
    label "process_high"
    label "publish_intermediate"
    container "${params.container__bracken}"

    input:
        path(kraken2_db)
        path(genomes_dir)
        path(nodes_dmp)
        path(names_dmp)
        val(read_len)

    output:
        path("bracken_db/database${read_len}mers.kmer_distrib"), emit: kmer_distrib

    shell:
    '''
    mkdir -p bracken_db/library bracken_db/taxonomy
    ln -s "$(readlink -f !{kraken2_db})"/*.k2d bracken_db/
    ln -s "$(readlink -f !{nodes_dmp})" bracken_db/taxonomy/nodes.dmp
    ln -s "$(readlink -f !{names_dmp})" bracken_db/taxonomy/names.dmp

    # One concatenated library file: bracken-build passes every library file to
    # kraken2 as its own argument, and ~31k genome paths exceed ARG_MAX.
    find -L !{genomes_dir} -name '*.fna.gz' -exec zcat {} + > bracken_db/library/library.fna

    grep '^>' bracken_db/library/library.fna \
        | sed -E 's/^>(kraken:taxid\\|([0-9]+)\\|[^ ]+).*/\\1\\t\\2/' \
        > bracken_db/seqid2taxid.map

    bracken-build -d bracken_db -t !{task.cpus} -k 35 -l !{read_len}
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
