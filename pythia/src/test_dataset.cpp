#include "Pythia8/Pythia.h"

#include <vector>
#include <iostream>

using namespace Pythia8;

int find_top_from_event(const Pythia8::Event& event){
    // Iterate through particles and analyze partons
    for(int i=0;i<event.size();i++){
        auto &p = event[i];

        if(p.id()!=6) continue;

        // Get top daughters and look for W+ and b
        int d1 = p.daughter1();
        int d2 = p.daughter2();

        if(event[d1].id()==6) continue; // Skip ISR; go until top decays

        for (int j=d1; j<=d2; j++){
            if(event[i].id()==5) return j; // If top decayed to b, return idx
        }
    }
    return 0; // Nothing good
}

int find_down_from_W(const Pythia8::Event& event, int W_idx){
    auto &Wboson = event[W_idx];

    int d1 = Wboson.daughter1();
    int d2 = Wboson.daughter2();

    if (event[d1].id()==24) return find_down_from_W(event, d1);

    for (int i=d1; d1<=d2; i++){
        if (event[i].idAbs()==1 || event[i].idAbs()==3){
            return i;
        }
    }
    return 0; // Nothing good
}

int find_down_from_top(const Pythia8::Event& event, int top_idx){
    auto &top = event[top_idx];

    int d1 = top.daughter1();
    int d2 = top.daughter2();

    for (int i=d1; d1<=d2; i++){
        // Find down type quark through W boson
        if (event[i].id()==24){
            return find_down_from_W(event, i);
        }
        // Find down quark directly from top
        if (event[i].idAbs()==1 || event[i].idAbs()==3){
            return i;
        }
    }
    return 0; // Nothing good
}

void traverse_history(const Pythia8::Event& event, std::vector<int> &fromDown, int current_idx){
    // Flag current particle as from down
    fromDown[current_idx] = 1;

    // Traverse Daughters
    int d1 = event[current_idx].daughter1();
    int d2 = event[current_idx].daughter2();

    std::cout << current_idx << "\t" << d1 << "\t" << d2 << std::endl;

    // Normal case where daughters are stored sequentially
    if (d1<=d2 && d1>0){
        for (int i=d1; d1<=d2; i++){
            return traverse_history(event, fromDown, i);
        }
    }

    // Special case where two daughters are stored not sequentially
    if (d2<d1 && d2<0){
        for (int i=0; i<2; i++){
            if (i==0) return traverse_history(event, fromDown, d1);
            if (i==1) return traverse_history(event, fromDown, d2);
        }
    }

    // There are no more daughters
    if ( (d1==0) && (d2==0) ) return;
}

std::vector<int> find_down_daughters(const Pythia8::Event& event, int down_idx){
    //Initialize fromDown vector
    std::vector<int> fromDown(event.size(), 0);

    // Traverse Down History
    traverse_history(event, fromDown, down_idx);

    return fromDown;
}

int main()
{
    Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = ../../madgraph/pp_tt_semi_full/Events/run_01/unweighted_events.lhe.gz");

    pythia.readString("Next:numberCount = 1");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    // Allow for possibility of a few faulty events.
    int nAbort = 10;
    int iAbort = 0;
    
    int nEvents = 1;
    int iEvents = 0;

    // Begin Event Loop; generate until none left in input file
    while (iEvents < nEvents) {

        // Generate events, and check whether generation failed.
        if (!pythia.next()) {
          // If failure because reached end of file then exit event loop.
          if (pythia.info.atEndOfFile()) break;
          ++iAbort;
          continue;
        }

        int top_idx = find_top_from_event(pythia.event);
        int down_idx = find_down_from_top(pythia.event, top_idx);
        
        auto &down = pythia.event[down_idx]; 

        //std::cout << down_idx << "\t" << down.id() << "\t" << down.status() << std::endl;

        std::vector<int> fromDown;
        fromDown = find_down_daughters(pythia.event, down_idx);

        for (int i=0; i<fromDown.size();i++){
            if (fromDown[i]==1) std::cout << i << std::endl;
        }

        /*
        // Iterate through particles and analyze partons
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];

            if(p.id()!=6) continue;

            // Initialize pointers
            auto &top = p;

            // Get top daughters and look for W+ and b
            int d1_idx = top.daughter1();
            int d2_idx = top.daughter2();

            if(pythia.event[d1_idx].id()==6) continue; // Skip ISR; go until top decays

            bool Wboson = false;
            bool bquark = false;
            bool upquark = false;
            bool downquark = false;

            for (int idx=d1_idx; idx<=d2_idx; idx++){
                if (pythia.event[idx].id()==24){
                    std::cout << "Found W!" << std::endl;
                    int down_idx = find_down_from_W(pythia.event,idx);
                }
                if (pythia.event[idx].id()==5){
                    std::cout << "Found b!" << std::endl;
                }
                if (pythia.event[idx].idAbs()==1 || pythia.event[idx].idAbs()==3){
                    std::cout << "Found d|s!" << std::endl;
                }
                if (pythia.event[idx].idAbs()==2 || pythia.event[idx].idAbs()==4){
                    std::cout << "Found u|c!" << std::endl;
                }
            }
            if( (d1.id()==24) || (d2.id()==24) ){
                std::cout << "Found t->Wq Decay!" << std::endl;

                if(d1.id()==24) {Wboson = d1;}
                if(d2.id()==24) {Wboson = d2;}

                auto &d1 = pythia.event[Wboson.daughter1()];
                auto &d2 = pythia.event[Wboson.daughter2()];

                if( (d1.idAbs()==1 || d1.idAbs()==3) || (d2.idAbs()==1 || d2.idAbs()==3) ){
                    std::cout << "Found W->qqbar Decay!" << std::endl;
                    if(d1.idAbs()==1 || d1.idAbs()==3) down = d1;
                    if(d2.idAbs()==1 || d2.idAbs()==3) down = d2;

                } // End down branching
            
            } // End top branching
            else {
                std::cout << "W NOT FOUND" << std::endl;
                //pythia.event.list();
                //return 0;
            }


        } // End particle event loop
        */
        iEvents++;
    } // End pythia event loop

    return 0;
}
