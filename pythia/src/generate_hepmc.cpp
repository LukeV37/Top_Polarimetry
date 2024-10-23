#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

using namespace Pythia8;

int main()
{
    Pythia pythia;
    //pythia.readFile("../cards/pythia8_card.dat");
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = ../../madgraph/pp_tt_semi_full/Events/run_01/unweighted_events.lhe.gz");
    Pythia8ToHepMC toHepMC("../output/pp_tt_semi_full.hepmc");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

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
    }

    return 0;
}
