#!/bin/bash

if [ -z "$7" ]; then
    echo "Must enter 7 arguments"
    echo -e "\t1: Dataset Tag e.g. unpolarized_10k"
    echo -e "\t2: Polarization: L, R, U"
    echo -e "\t3: Generation: inclusive, first, second"
    echo -e "\t4: Number of Runs"
    echo -e "\t5: Number of Events per Run"
    echo -e "\t6: Max num cpu cores"
    echo -e "\t7: Initial seed"
    exit 1
fi

dataset_tag=$1
polarization=$2
generation=$3
num_runs=$4
num_events_per_run=$5
max_cpu_cores=$6
seed=$7

set -e

###########################
### MadGraph Generation ###
###########################

# Edit process card
sed "s/output pp_tt_semi_full/output pp_tt_semi_full_${dataset_tag}/" proc_card_mg5.dat > proc_card.tmp

# Apply polarization (U requires no spin tag)
if [ "$polarization" = "L" ]; then
  sed -i "s/p p > t t~/p p > t{L} t~/" proc_card.tmp
elif [ "$polarization" = "R" ]; then
  sed -i "s/p p > t t~/p p > t{R} t~/" proc_card.tmp
fi

# Apply generation (inclusive requires no change)
if [ "$generation" = "first" ]; then
  sed -i "s/t > b j j/t > b u d~/" proc_card.tmp
elif [ "$generation" = "second" ]; then
  sed -i "s/t > b j j/t > b c s~/" proc_card.tmp
fi

sed "s/multi_run.*/multi_run $num_runs/" multi_run.config > multi_run.tmp
sed -i "s/set nevents.*/set nevents $num_events_per_run/" multi_run.tmp
sed -i "s/set iseed.*/set iseed $seed/" multi_run.tmp

# Run mg5_aMC binary on the process card
../submodules/mg5amcnlo-v3.5.5/bin/mg5_aMC proc_card.tmp

# Copy the cuts.f card to the SubProcesses folder
cp ./cuts.f "./pp_tt_semi_full_${dataset_tag}/SubProcesses/"

echo "Please be patient while MadGraph generates processes..."
"./pp_tt_semi_full_${dataset_tag}/bin/madevent" multi_run.tmp | tee "MadGraph_${dataset_tag}.log"

# Clean up workspace
rm -f py.py MG5_debug ME5_debug
rm *.tmp

echo
echo -e "\tMadGraph Generation Done!"
