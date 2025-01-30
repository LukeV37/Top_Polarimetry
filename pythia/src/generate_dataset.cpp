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
#include "include/trace_origin_top.h"

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

    std::vector<std::vector<float>> jet_trk_pT, jet_trk_eta, jet_trk_phi, jet_trk_q, jet_trk_d0, jet_trk_z0;
    std::vector<std::vector<int>> jet_trk_origin, jet_trk_pid;
    FastJet->Branch("jet_trk_pt", &jet_trk_pT);
    FastJet->Branch("jet_trk_eta", &jet_trk_eta);
    FastJet->Branch("jet_trk_phi", &jet_trk_phi);
    FastJet->Branch("jet_trk_q", &jet_trk_q);
    FastJet->Branch("jet_trk_d0", &jet_trk_d0);
    FastJet->Branch("jet_trk_z0", &jet_trk_z0);
    FastJet->Branch("jet_trk_pid", &jet_trk_pid);
    FastJet->Branch("jet_trk_origin", &jet_trk_origin);

    std::vector<float> trk_pT, trk_eta, trk_phi, trk_q, trk_d0, trk_z0;
    std::vector<int> trk_origin, trk_pid;
    FastJet->Branch("trk_pt", &trk_pT);
    FastJet->Branch("trk_eta", &trk_eta);
    FastJet->Branch("trk_phi", &trk_phi);
    FastJet->Branch("trk_q", &trk_q);
    FastJet->Branch("trk_d0", &trk_d0);
    FastJet->Branch("trk_z0", &trk_z0);
    FastJet->Branch("trk_pid", &trk_pid);
    FastJet->Branch("trk_origin", &trk_origin);

    // Configure Jet parameters
    float pTmin_jet = 250; // GeV
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

        // Initialize vector for fastjet clustering and particle index
        std::vector<fastjet::PseudoJet> fastjet_particles;
        int particle_num=0;

        // prepare for filling
        jet_pt.clear(); jet_eta.clear(); jet_phi.clear(); jet_m.clear();
        jet_trk_pT.clear(); jet_trk_eta.clear(); jet_trk_phi.clear(); jet_trk_q.clear(); jet_trk_d0.clear(); jet_trk_z0.clear(); jet_trk_origin.clear(); jet_trk_pid.clear();
        trk_pT.clear(); trk_eta.clear(); trk_phi.clear(); trk_q.clear(); trk_d0.clear(); trk_z0.clear(); trk_origin.clear(); trk_pid.clear();


        // Loop through particles in the event
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];
            particle_num++; // Keep track of particle num
            
            // Do not consider intermediate particles for clustering
            if (not p.isFinal()) continue;
            // Do not consider neutrinos in clustering
            if (std::abs(p.id())==12 || std::abs(p.id())==14 || std::abs(p.id())==16) continue;

            // Convert particles to PseduoJet object, set the user idx, and append to the list of fastjet particles
            fastjet::PseudoJet fj(p.px(), p.py(), p.pz(), p.e());
            fj.set_user_index(particle_num-1); // Subtract 1 to become 0 based
            fastjet_particles.push_back(fj);

            // Fill trk vector with all fastjet candidates
            trk_pT.push_back(p.pT());
            trk_eta.push_back(p.eta());
            trk_phi.push_back(p.phi());
            trk_q.push_back(p.charge());
            double d0,z0; find_ip(p.pT(),p.eta(),p.phi(),p.xProd(),p.yProd(),p.zProd(),d0,z0);
            trk_d0.push_back(d0);
            trk_z0.push_back(z0);
            int bcflag = 0;
            int origin = trace_origin_top(pythia.event,j,bcflag);
            trk_origin.push_back(origin);
            trk_pid.push_back(p.id());
        }

        // Cluster particles using fastjet
        fastjet::ClusterSequence clustSeq(fastjet_particles, jetDefs["fatjet"]);
        auto jets = fastjet::sorted_by_pt( clustSeq.inclusive_jets(pTmin_jet) );

        // Loop through clustered jets
        for (auto jet:jets) {
            jet_pt.push_back(jet.pt()); jet_eta.push_back(jet.eta()); jet_phi.push_back(jet.phi()); jet_m.push_back(jet.m());

            // Temporary vectors with jet constituent info
            std::vector<float> jet_trk_pT_tmp, jet_trk_eta_tmp, jet_trk_phi_tmp, jet_trk_q_tmp, jet_trk_d0_tmp, jet_trk_z0_tmp;
            std::vector<int> jet_trk_origin_tmp, jet_trk_pid_tmp;

            // Loop through jet constituents
            for (auto trk:jet.constituents()) {
                int idx = trk.user_index();
                auto &p = pythia.event[idx];
                jet_trk_pT_tmp.push_back(p.pT());
                jet_trk_eta_tmp.push_back(p.eta());
                jet_trk_phi_tmp.push_back(p.phi());
                jet_trk_q_tmp.push_back(p.charge());
                double d0,z0; find_ip(p.pT(),p.eta(),p.phi(),p.xProd(),p.yProd(),p.zProd(),d0,z0);
                jet_trk_d0_tmp.push_back(d0);
                jet_trk_z0_tmp.push_back(z0);

                int bcflag = 0;
                int origin = trace_origin_top(pythia.event,idx,bcflag);
                jet_trk_origin_tmp.push_back(origin);
                jet_trk_pid_tmp.push_back(p.id());

            } // End loop through trks

            jet_trk_pT.push_back(jet_trk_pT_tmp);
            jet_trk_eta.push_back(jet_trk_eta_tmp);
            jet_trk_phi.push_back(jet_trk_phi_tmp);
            jet_trk_q.push_back(jet_trk_q_tmp);
            jet_trk_d0.push_back(jet_trk_d0_tmp);
            jet_trk_z0.push_back(jet_trk_z0_tmp);
            jet_trk_origin.push_back(jet_trk_origin_tmp);
            jet_trk_pid.push_back(jet_trk_pid_tmp);

        } // End loop through jets

        FastJet->Fill();

    } // End pythia event loop

    output->Write();
    output->Close();

    return 0;
}
