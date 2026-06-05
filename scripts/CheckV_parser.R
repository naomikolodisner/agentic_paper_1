#!/usr/bin/env Rscript

# Load necessary libraries
library(data.table)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check for correct usage
if (length(args) < 3) {
  stop("Usage: CheckV_parser.R -i <input_file> -l <length_threshold> -o <output_file>")
}

# Parse arguments
input_file <- NULL
length_threshold <- NULL
output_file <- NULL



for (i in seq(1, length(args), by = 2)) {
  if (args[i] == "-i") {
    input_file <- args[i + 1]
  } else if (args[i] == "-l") {
    length_threshold <- as.numeric(args[i + 1])
  } else if (args[i] == "-o") {
    output_file <- args[i + 1]
  }
}

# Validate arguments
if (is.null(input_file) || is.null(length_threshold) || is.null(output_file)) {
  stop("Missing required arguments. Ensure input file, length threshold, and output file are provided.")
}

# Read the contamination.tsv file
checkv_data <- fread(input_file)

# Ensure necessary columns exist
if (!("contig_id" %in% colnames(checkv_data) && "contig_length" %in% colnames(checkv_data))) {
  stop("Input file must contain 'contig_id' and 'contig_length' columns.")
}

# Filter contigs based on length threshold
filtered_contigs <- checkv_data[contig_length >= length_threshold, .(contig_id)]

# Write the filtered contig IDs to the output CSV
fwrite(filtered_contigs, file = output_file, col.names = TRUE)

cat("Filtered contig IDs written to:", output_file, "\n")
