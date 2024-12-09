#!/bin/bash

# Validate RDF output using Apache Jena's riot tool
if [ $# -ne 1 ]; then
    echo "Usage: $0 <rdf_file>"
    exit 1
fi

riot --validate $1
