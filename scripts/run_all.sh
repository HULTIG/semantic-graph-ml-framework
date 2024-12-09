#!/bin/bash

# Generate all RDF outputs for each pilot
for pilot in source_a souce_b; do
    ./generate_rdf.sh rml/$pilot/$pilot_rml.ttl rdf/$pilot/$pilot_rdf.ttl
    ./generate_rdf.sh rml/$pilot/$pilot_rml-imp.ttl rdf/$pilot/$pilot_rdf-imp.ttl
    ./generate_rdf.sh rml/unified/$pilot/$pilot_rml-unified.ttl rdf/unified/$pilot/$pilot_rdf-unified.ttl
done
