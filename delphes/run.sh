#!/bin/bash
echo "Converting LHE to ROOT file. Please be patient..."
python3 LHEReader.py --input ../madgraph/pp_tt_semi_full/Events/run_01/unweighted_events.lhe --output partons.root
echo "Running DELPHES simluation. Please be patient..."
../submodules/delphes-v3.5.0/DelphesHepMC2 delphes_card_HLLHC.tcl dataset.root ../madgraph/pp_tt_semi_full/Events/run_01/tag_1_pythia8_events.hepmc
