#include <iostream>
#include <vector>

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include <TRandom.h>
#include "TString.h"

#include "include/estimate_ip.h"
#include "include/trace_origin.h"

int main()
{
    // Initialize Pythia Settings
    Pythia8::Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = ../../madgraph/pp_tt_semi_full/Events/run_01/unweighted_events.lhe.gz");
    Pythia8::Pythia8ToHepMC toHepMC("../shower.hepmc");
    pythia.readString("Next:numberCount = 100");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    // Initialize output ROOT file, TTree, and Branches
    TFile *output = new TFile("../dataset.root","recreate");
    TTree *FastJet = new TTree("fastjet", "fastjet");

    std::vector<float> jet_pt, jet_eta, jet_phi, jet_m;
    FastJet->Branch("jet_pt", &jet_pt);
    FastJet->Branch("jet_eta", &jet_eta);
    FastJet->Branch("jet_phi", &jet_phi);
    FastJet->Branch("jet_m", &jet_m);

    std::vector<std::vector<float>> trk_pT, trk_eta, trk_phi, trk_m, trk_q, trk_d0, trk_z0;
    std::vector<std::vector<int>> trk_isTop;
    FastJet->Branch("trk_pT", &trk_pT);
    FastJet->Branch("trk_eta", &trk_eta);
    FastJet->Branch("trk_phi", &trk_phi);
    FastJet->Branch("trk_m", &trk_m);
    FastJet->Branch("trk_q", &trk_q);
    FastJet->Branch("trk_d0", &trk_d0);
    FastJet->Branch("trk_z0", &trk_z0);
    FastJet->Branch("trk_isTop", &trk_isTop);

    // Configure Jet parameters
    float pTmin_jet = 25; // GeV
    std::map<TString, fastjet::JetDefinition> jetDefs;
    jetDefs["anti-kt"] = fastjet::JetDefinition(
      fastjet::antikt_algorithm, 0.4, fastjet::E_scheme, fastjet::Best);
    jetDefs["kt"] = fastjet::JetDefinition(
      fastjet::kt_algorithm, 0.4, fastjet::E_scheme, fastjet::Best);
    jetDefs["CA"] = fastjet::JetDefinition(
      fastjet::cambridge_algorithm, 0.4, fastjet::E_scheme, fastjet::Best);
    jetDefs["fatjet"] = fastjet::JetDefinition(
      fastjet::cambridge_algorithm, 1.0, fastjet::E_scheme, fastjet::Best);

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

        // Initialize vector for fastjet clustering and user id
        std::vector<fastjet::PseudoJet> fastjet_particles;
        int user_id=0;

        // Loop through particles in the event
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];
            
            // Only consider final state particles for clustering
            if (not p.isFinal()) continue;
            // Do not consider neutrinos in clustering
            if (std::abs(p.id())==12 || std::abs(p.id())==14 || std::abs(p.id())==16) continue;

            // Convert particles to PseduoJet object, set the user idx, and append to the list of fastjet particles
            fastjet::PseudoJet fj(p.px(), p.py(), p.pz(), p.e());
            fj.set_user_index(user_id++);
            fastjet_particles.push_back(fj);
        }

        // prepare for filling
        jet_pt.clear(); jet_eta.clear(); jet_phi.clear(); jet_m.clear();
        trk_pT.clear(); trk_eta.clear(); trk_phi.clear(); trk_m.clear(); trk_q.clear(); trk_d0.clear(); trk_z0.clear(); trk_isTop.clear();

        // Cluster particles using fastjet
        fastjet::ClusterSequence clustSeq(fastjet_particles, jetDefs["fatjet"]);
        auto jets = fastjet::sorted_by_pt( clustSeq.inclusive_jets(pTmin_jet) );

        // Loop through clustered jets
        for (auto jet:jets) {
            jet_pt.push_back(jet.pt()); jet_eta.push_back(jet.eta()); jet_phi.push_back(jet.phi()); jet_m.push_back(jet.m());

            // Temporary vectors with jet constituent info
            std::vector<float> trk_pT_tmp, trk_eta_tmp, trk_phi_tmp, trk_m_tmp, trk_q_tmp, trk_d0_tmp, trk_z0_tmp;
            std::vector<int> trk_isTop_tmp; 

            // Loop through jet constituents
            for (auto trk:jet.constituents()) {
                int idx = trk.user_index();
                auto &p = pythia.event[idx];
                trk_pT_tmp.push_back(p.pT());
                trk_eta_tmp.push_back(p.eta());
                trk_phi_tmp.push_back(p.phi());
                trk_m_tmp.push_back(p.m());
                trk_q_tmp.push_back(p.charge());
                double d0,z0; find_ip(p.pT(),p.eta(),p.phi(),p.xProd(),p.yProd(),p.zProd(),d0,z0);
                trk_d0_tmp.push_back(d0);
                trk_z0_tmp.push_back(z0);

                int bcflag = 0;
                int isTop = trace_origin_top(pythia.event,idx,bcflag);
                trk_isTop_tmp.push_back(isTop);

            } // End loop through trks

            trk_pT.push_back(trk_pT_tmp);
            trk_eta.push_back(trk_eta_tmp);
            trk_phi.push_back(trk_phi_tmp);
            trk_m.push_back(trk_m_tmp);
            trk_q.push_back(trk_q_tmp);
            trk_d0.push_back(trk_d0_tmp);
            trk_z0.push_back(trk_z0_tmp);
            trk_isTop.push_back(trk_isTop_tmp);

        } // End loop through jets

        FastJet->Fill();

    } // End pythia event loop

    output->Write();
    output->Close();

    return 0;
}
