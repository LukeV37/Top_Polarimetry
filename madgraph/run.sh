#!/bin/bash

# Run mg5_aMC binary on the process card
../submodules/mg5amcnlo-v3.5.5/bin/mg5_aMC cards/proc_card_mg5.dat

# Copy the cuts.f card to the SubProcesses folder
cp ./include/cuts.f ./pp_tt_semi_full/SubProcesses/

# Copy the run card to the Cards folder
cp ./cards/run_card.dat ./pp_tt_semi_full/Cards/

# Generate the Events!
./pp_tt_semi_full/bin/generate_events

# Clean up workspace (generated automatically by madgraph binary)
rm -f py.py
