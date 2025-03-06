#!/bin/bash

if [ -z "$4" ]; then
    echo "Must enter 4 agruments"
    echo -e "\t1: Dataset Tag e.g. unpolarized_10k"
    echo -e "\t2: Polarization: L, R, U"
    echo -e "\t3: Number of Runs"
    echo -e "\t4: Number of Events per Run"
    exit 1
fi

dataset_tag=$1
polarization=$2
num_runs=$3
num_events_per_run=$4

# Error handling before launching event generation
set -e
python -c "import ROOT; import uproot; import awkward"

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
if [ "$polarization" = "down" ]; then
  sed -i "s/generate p p > t t~, t > b j j, t~ > b~ l- vl~/generate p p > t t~, t > b u d~, t~ > b~ l- vl~/" proc_card.tmp
fi
if [ "$polarization" = "charm" ]; then
  sed -i "s/generate p p > t t~, t > b j j, t~ > b~ l- vl~/generate p p > t t~, t > b c s~, t~ > b~ l- vl~/" proc_card.tmp
fi

# Edit run card
#sed "s/\(.*\)= nevents\(.*\)/ $num_events = nevents\2/" include/run_card.dat > run_card.tmp
sed "s/multi_run.*/multi_run $num_runs/" include/multi_run.config > multi_run.tmp
sed -i "s/set nevents.*/set nevents $num_events_per_run/" multi_run.tmp

# Run mg5_aMC binary on the process card
../submodules/mg5amcnlo-v3.5.5/bin/mg5_aMC proc_card.tmp

# Copy the cuts.f card to the SubProcesses folder
cp ./include/cuts.f "./pp_tt_semi_full_${dataset_tag}/SubProcesses/"

# Copy the cards to the Cards folder
#cp run_card.tmp "./pp_tt_semi_full_${dataset_tag}/Cards/run_card.dat"

# Generate the Events!
#"./pp_tt_semi_full_${dataset_tag}/bin/generate_events" -f
echo "Please be patient while MadGraph generates processes..."
"./pp_tt_semi_full_${dataset_tag}/bin/madevent" multi_run.tmp | tee "MadGraph_${dataset_tag}.log"

# Clean up workspace (generated automatically by madgraph binary)
rm -f py.py
rm -f MG5_debug
rm -f ME5_debug
rm *.tmp

#######################
### Post Processing ###
#######################

function calc_labels {
    tag=$1
    run=$2

    echo -e "\tCalc Labels for Run: $run"
    # Extract LHE file with gzip
    #echo -e "\tDecompressing lhe file..."
    gzip -dk "./pp_tt_semi_full_${tag}/Events/run_01_$run/unweighted_events.lhe.gz"

    # Find line with version number and delete the proceeding warning
    # This is needed since I am using git tag instead of production version
    line_num=$(awk '/VERSION 3.5.5/ {print NR}' "pp_tt_semi_full_${tag}/Events/run_01_$run/unweighted_events.lhe")
    start_line=$((line_num+1))
    end_line=$((line_num+4))
    sed -i "${start_line},${end_line}d" "pp_tt_semi_full_${tag}/Events/run_01_$run/unweighted_events.lhe"

    # Now that warning message is removed, use LHEReader.py to convert LHE file to root file
    #echo -e "\tConverting lhe file to root format..."
    python include/LHEReader.py --input "pp_tt_semi_full_${tag}/Events/run_01_$run/unweighted_events.lhe" --output "pp_tt_semi_full_${tag}/hard_process_${tag}_$i.root"

    # Calculate labels
    python include/TLorentz_Labels.py $tag $run

    # Clean workspace (uncompressed version no longer needed)
    rm -f "./pp_tt_semi_full_${tag}/Events/run_01_$run/unweighted_events.lhe"

    echo -e "\tDone Calc Labels for Run: $run"
}

for (( i=0 ; i<$num_runs ; i++ ));
do
    calc_labels $dataset_tag $i &
done
wait

echo
echo -e "\tMadGraph Generation Done!"
