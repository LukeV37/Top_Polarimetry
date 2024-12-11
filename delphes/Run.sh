#!/bin/bash

for cut in 200 350 500;
do
    echo "Converting LHE to ROOT file. Please be patient..."
    python3 LHEReader.py --input ../madgraph/minpTt_$cut/Events/run_01/unweighted_events.lhe --output partons_$cut.root
    echo "Running DELPHES simluation. Please be patient..."
    sed -i "1082s/  set JetPTMin \(.*\)/  set JetPTMin ${cut}/" delphes_card_HLLHC.tcl
    ../submodules/delphes-v3.5.0/DelphesHepMC2 delphes_card_HLLHC.tcl dataset_$cut.root ../madgraph/minpTt_$cut/Events/run_01/tag_1_pythia8_events.hepmc
done

echo "Done!"
