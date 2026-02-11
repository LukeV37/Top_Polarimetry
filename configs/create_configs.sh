#!/bin/bash

# Define the arrays of values to iterate over
processes=("U" "L" "R")
R_values=("1.0" "1.5" "2.0")
pT_values=("250" "400" "550")

# Template file
template="job.config"

# Loop over all combinations
for process in "${processes[@]}"; do
    for R in "${R_values[@]}"; do
        for pT in "${pT_values[@]}"; do
            # Create a filename for this configuration
            config_file="job_${process}_R${R/./_}_pT${pT}.config"
            
            echo "Generating: $config_file"
            
            # Copy template and replace placeholders
            sed -e "s/__PROCESS__/$process/g" \
                -e "s/__R__/$R/g" \
                -e "s/__pT__/$pT/g" \
                "$template" > "configs/$config_file"
            
        done
    done
done

echo "All config files generated!"
