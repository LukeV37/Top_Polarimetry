#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include "TFile.h"
#include "TTree.h"

#include <vector>

using namespace Pythia8;

int main()
{
    Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = ../../madgraph/pp_tt_semi_full/Events/run_01/unweighted_events.lhe.gz");
    Pythia8ToHepMC toHepMC("../output/pp_tt_semi_full.hepmc");

    pythia.readString("Next:numberCount = 1");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    // Initialize output ROOT file, TTree, and Branches
    TFile *output = new TFile("../output/dataset.root","recreate");
    TTree *Pythia = new TTree("pythia", "pythia");
    std::vector<float> px, py, pz, e, m;
    std::vector<int> id;
    Pythia->Branch("px", &px);
    Pythia->Branch("py", &py);
    Pythia->Branch("pz", &pz);
    Pythia->Branch("e", &e);
    Pythia->Branch("id", &id);

    // Allow for possibility of a few faulty events.
    int nAbort = 10;
    int iAbort = 0;

    // Begin Event Loop; generate until none left in input file
    while (iAbort < nAbort) {

        // Generate events, and check whether generation failed.
        if (!pythia.next()) {
          // If failure because reached end of file then exit event loop.
          if (pythia.info.atEndOfFile()) break;
          ++iAbort;
          continue;
        }

        // Write out event to a hepmc file
        toHepMC.writeNextEvent( pythia );

        // Iterate through particles and analyze partons
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];

            if(p.id()!=6) continue;

            // Initialize pointers
            auto &top = p;
            auto &Wboson = p;
            auto &down = p;

            // Get top daughters and look for W+ and b
            auto &d1 = pythia.event[top.daughter1()];
            auto &d2 = pythia.event[top.daughter2()];
            if(d1.id()==6) continue;
            if( (d1.id()==24) || (d2.id()==24) ){
                std::cout << "Found t->Wq Decay!" << std::endl;

                if(d1.id()==24) {Wboson = d1; std::cout<< d2.id() << std::endl;}
                if(d2.id()==24) {Wboson = d2; std::cout<< d1.id() << std::endl;}



                // Store top quark kinematics
                px.push_back(top.px());
                py.push_back(top.py());
                pz.push_back(top.pz());
                e.push_back(top.e());
                m.push_back(top.m());
                id.push_back(top.id());

                auto &d1 = pythia.event[Wboson.daughter1()];
                auto &d2 = pythia.event[Wboson.daughter2()];

                if( (d1.idAbs()==1 || d1.idAbs()==3) || (d2.idAbs()==1 || d2.idAbs()==3) ){
                    std::cout << "Found W->qqbar Decay!" << std::endl;
                    if(d1.idAbs()==1 || d1.idAbs()==3) down = d1;
                    if(d2.idAbs()==1 || d2.idAbs()==3) down = d2;

                    // Store down quark kinematics
                    px.push_back(down.px());
                    py.push_back(down.py());
                    pz.push_back(down.pz());
                    e.push_back(down.e());
                    m.push_back(down.m());
                    id.push_back(down.id());
                    
                } // End down branching

            } // End top branching
            else {pythia.event.list(); return 0;}

            Pythia->Fill();
            px.clear();
            py.clear();
            pz.clear();
            e.clear();
            m.clear();
            id.clear();

        } // End particle event loop

    } // End pythia event loop

    output->Write();
    output->Close();

    return 0;
}
