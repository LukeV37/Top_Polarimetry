#include "Pythia8/Pythia.h"

#include "TFile.h"
#include "TH1.h"

#include <vector>
#include <iostream>

#include "traverse_history.h"

using namespace Pythia8;

int main()
{
    Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = ../../../madgraph/pp_tt_semi_full/Events/run_01/unweighted_events.lhe.gz");
    pythia.readString("Next:numberCount = 100");

    TFile f("histo.root","RECREATE");
    TH1* h1 = new TH1I("d", "Num Daughters from Down Quark", 200, 0.0, 200);

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    int nEvent = 1000;
    int iEvent = 0;

    // Begin Event Loop; generate until none left in input file
    while (iEvent < nEvent) {

        // Generate events, and check whether generation failed.
        if (!pythia.next()) {
          // If failure because reached end of file then exit event loop.
          if (pythia.info.atEndOfFile()) break;
          continue;
        }

        int top_idx = find_top_from_event(pythia.event);
        int down_idx = find_down_from_top(pythia.event, top_idx);
        auto &down = pythia.event[down_idx]; 
        //std::cout << down_idx << "\t" << down.id() << "\t" << down.status() << std::endl;

        std::vector<int> fromDown;
        fromDown = find_down_daughters(pythia.event, down_idx);

        int num_daughters=0;
        for (int i=0;i<fromDown.size();i++){
            if (fromDown[i]==1) num_daughters++;
        }
        //std::cout << "Down Quark: num_daughters=" << num_daughters << std::endl;
        h1->Fill(num_daughters);

        if (num_daughters<7){
            std::cout << down_idx << "\t" << down.id() << "\t" << down.status() << std::endl;
            pythia.event.list();
        }

        iEvent++;

    } // End pythia event loop

    h1->Write();

    return 0;
}
