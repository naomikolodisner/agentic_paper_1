#!/usr/bin/env bash
# check_kraken2_accessions.sh
#
# Cross-checks accession IDs from the virfinder/inphared overlap list
# against the Kraken2 seqid2taxid.map.
#
# Usage:
#   bash check_kraken2_accessions.sh [accession_list] [seqid2taxid_map] [output_found] [output_missing]
#
# Defaults match the original database paths used in this analysis.

ACCESSION_LIST="${1:-/rs1/shares/brc/databases/inphared/in_inphared_and_virfinder_pre2014}"
SEQID2TAXID="${2:-/rs1/shares/brc/admin/databases/kraken2_pluspfp/seqid2taxid.map}"
OUTPUT_FOUND="${3:-accessions_in_kraken2.txt}"
OUTPUT_MISSING="${4:-accessions_missing_from_kraken2.txt}"

if [[ ! -f "$ACCESSION_LIST" ]]; then
    echo "ERROR: accession list not found: $ACCESSION_LIST" >&2
    exit 1
fi
if [[ ! -f "$SEQID2TAXID" ]]; then
    echo "ERROR: seqid2taxid.map not found: $SEQID2TAXID" >&2
    exit 1
fi

found=0
missing=0
> "$OUTPUT_FOUND"
> "$OUTPUT_MISSING"

while IFS= read -r acc; do
    [[ -z "$acc" ]] && continue
    if grep -q "|${acc}\." "$SEQID2TAXID"; then
        echo "$acc" >> "$OUTPUT_FOUND"
        ((found++))
    else
        echo "$acc" >> "$OUTPUT_MISSING"
        ((missing++))
    fi
done < "$ACCESSION_LIST"

total=$((found + missing))

echo "Checked $total accessions against: $SEQID2TAXID"
echo "  Found:   $found  -> $OUTPUT_FOUND"
echo "  Missing: $missing -> $OUTPUT_MISSING"

if [[ $missing -gt 0 ]]; then
    echo ""
    echo "Missing accessions:"
    cat "$OUTPUT_MISSING"
fi
