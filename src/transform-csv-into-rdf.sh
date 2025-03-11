#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Create symlinks to source files
for file in src/data/*.csv; do
    base_name=$(basename "$file")
    ln -sf "../$file" "data/$base_name"
done

# Array of RML files to process
rml_files=(
    "studies"
    # "outcomes"
    # "designs"
    # "outcome_analyses"
    # "eligibilities"
    # "interventions"
    # "interventions_studies"
    # "reported_event_totals"
    # "drop_withdrawals"
    # "facilities"
    # "sponsors_studies"
    # "conditions_studies"
    # "conditions"
    # "sponsors"
    # "facilities_studies"
)

# Create output directory if it doesn't exist
mkdir -p output

# Process each RML file
for file in "${rml_files[@]}"; do
    echo "Processing $file..."
    java -jar src/data/rmlmapper-6.5.1-r371-all.jar -m "src/data/enhanced-rml/mappings/$file.rml.ttl" -o "output/$file-rdf.ttl" -s turtle
    if [ $? -eq 0 ]; then
        echo "✓ Successfully transformed $file"
    else
        echo "✗ Error transforming $file"
    fi
done

echo "All transformations completed. Check the output directory for results." 