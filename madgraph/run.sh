#!/bin/bash

if [ -z "$3" ]; then
    echo "Must enter 3 agruments"
    echo "1: Dataset Tag e.g. unpolarized_10k"
    echo "2: Polarization: L, R, U"
    echo "3: Number of Events"
    exit 1
fi

dataset_tag=$1
polarization=$2
num_events=$3

###########################
### MadGraph Generation ###
###########################

# Edit process card
sed "s/output pp_tt_semi_full/output pp_tt_semi_full_${dataset_tag}/" include/proc_card_mg5.dat > proc_card.tmp
if [ "$polarization" = "L" ]; then
  sed -i "s/generate p p > t t~, t > b j j, t~ > b~ l- vl~/generate p p > t{L} t~, t > b j j, t~ > b~ l- vl~/" proc_card.tmp
fi
if [ "$polarization" = "R" ]; then
  sed -i "s/generate p p > t t~, t > b j j, t~ > b~ l- vl~/generate p p > t{R} t~, t > b j j, t~ > b~ l- vl~/" proc_card.tmp
fi

# Edit run card
sed "s/\(.*\)= nevents\(.*\)/ $num_events = nevents\2/" include/run_card.dat > run_card.tmp

# Run mg5_aMC binary on the process card
../submodules/mg5amcnlo-v3.5.5/bin/mg5_aMC proc_card.tmp

# Copy the cuts.f card to the SubProcesses folder
cp ./include/cuts.f "./pp_tt_semi_full_${dataset_tag}/SubProcesses/"

# Copy the cards to the Cards folder
cp run_card.tmp "./pp_tt_semi_full_${dataset_tag}/Cards/run_card.dat"

# Generate the Events!
"./pp_tt_semi_full_${dataset_tag}/bin/generate_events"

# Clean up workspace (generated automatically by madgraph binary)
rm -f py.py
rm *.tmp

#######################
### Post Processing ###
#######################

# Extract LHE file with gzip
echo "Decompressing lhe file..."
gzip -dk "./pp_tt_semi_full_${dataset_tag}/Events/run_01/unweighted_events.lhe.gz"

# Find line with version number and delete the proceeding warning 
# This is needed since I am using git tag instead of production version
line_num=$(awk '/VERSION 3.5.5/ {print NR}' "pp_tt_semi_full_${dataset_tag}/Events/run_01/unweighted_events.lhe")
start_line=$((line_num+1))
end_line=$((line_num+4))
sed -i "${start_line},${end_line}d" "pp_tt_semi_full_${dataset_tag}/Events/run_01/unweighted_events.lhe"

# Now that warning message is removed, use LHEReader.py to convert LHE file to root file
echo "Converting lhe file to root format..."
python3 include/LHEReader.py --input "pp_tt_semi_full_${dataset_tag}/Events/run_01/unweighted_events.lhe" --output "hard_process_${dataset_tag}.root"

# Calculate labels
python3 include/TLorentz_Labels.py $dataset_tag

# Clean workspace (uncompressed version no longer needed)
rm -f "./pp_tt_semi_full_${dataset_tag}/Events/run_01/unweighted_events.lhe"
