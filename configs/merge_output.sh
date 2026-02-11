#!/bin/bash

cd ../pythia

for dir in WS_*/; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        (cd "$dir" && hadd -f combined.root dataset*.root)
        echo ""
    fi
done

cd ../configs
