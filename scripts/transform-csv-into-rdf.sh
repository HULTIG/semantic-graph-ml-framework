#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# Array of RML files to process
rml_files=(
    "studies"
    "outcomes"
    "interventions"
    "reported_event_totals"
    "drop_withdrawals"
    "facilities"
    "sponsors_studies"
    "conditions_studies"
)

# Process each RML file
for file in "${rml_files[@]}"; do
    echo "Processing $file..."
    java -jar data/rmlmapper-6.5.1-r371-all.jar -m "data/$file.rml.ttl" -o "output/$file-rdf.ttl" -s turtle
    if [ $? -eq 0 ]; then
        echo "✓ Successfully transformed $file"
    else
        echo "✗ Error transforming $file"
    fi
done

echo "All transformations completed. Check the output directory for results." 