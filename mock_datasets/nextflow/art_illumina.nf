#!/usr/bin/env nextflow

// Simulate paired-end reads with ART Illumina, each genome in a multi-FASTA at its
// own fold coverage, pooled into one R1/R2 pair.
// coverages is comma-separated, in the FASTA's record order (one conditions.tsv row
// from 09_pick_refs.py, e.g. "1,0.5"). read_len and art_profile come from the
// background the reads will be mixed into (ART_PROFILE_BY_MODEL in config.py).
// Read IDs keep ART's "<genome_id>-<n>/1" naming, which the kraken2 check relies on.
process art_illumina {
    label "process_low"
    label "publish_intermediate"
    container "${params.container__art}"

    input:
        tuple val(sample_id), path(contigs), val(coverages), val(read_len), val(art_profile)

    output:
        tuple val(sample_id), path("${sample_id}_viral.1.fq"), path("${sample_id}_viral.2.fq"), emit: reads

    shell:
    '''
    ids=( $(grep '^>' !{contigs} | sed 's/^>//; s/[[:space:]].*//') )
    IFS=',' read -ra covs <<< "!{coverages}"
    if [ "${#covs[@]}" -ne "${#ids[@]}" ]; then
        echo "ERROR: ${#covs[@]} coverages for ${#ids[@]} genomes in !{contigs}" >&2
        exit 1
    fi

    # One FASTA per genome, since ART applies a single -f to every record in -i
    awk '/^>/ { if (out) close(out); out = substr($1, 2) ".fa" } { print > out }' !{contigs}

    for i in "${!ids[@]}"; do
        art_illumina \
        -ss !{art_profile} \
        -p \
        -na \
        -i "${ids[$i]}.fa" \
        -f "${covs[$i]}" \
        -l !{read_len} \
        -m 200 \
        -s 10 \
        -o "art_${ids[$i]}_"
    done

    cat art_*_1.fq > !{sample_id}_viral.1.fq
    cat art_*_2.fq > !{sample_id}_viral.2.fq
    '''
}
