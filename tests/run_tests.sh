#!/bin/bash

# Validate all RDF outputs
for rdf_file in rdf/**/*.ttl; do
    echo "Validating $rdf_file"
    ./validation/validate_rdf.sh $rdf_file
done
