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
#include "include/traverse_history.h"

int main(int argc, char *argv[])
{
    if (argc < 2){
        std::cout << "Error! Must enter 2 arguments" << std::endl;
        std::cout << "1: Dataset Tag (from MadGraph)" << std::endl;
        std::cout << "2: Number of Runs (from MadGraph)" << std::endl;
        return 1;
    }
    char *dataset_tag = argv[1];
    char *run_num = argv[2];

    std::string inputFile = std::string("../../madgraph/pp_tt_semi_full_")+std::string(dataset_tag)+std::string("/Events/run_01_")+std::string(run_num)+std::string("/unweighted_events.lhe.gz");

    // Initialize Pythia Settings
    Pythia8::Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = "+inputFile);
    //Pythia8::Pythia8ToHepMC toHepMC("../shower.hepmc");
    pythia.readString("Next:numberCount = 1000");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    // Initialize output ROOT file, TTree, and Branches
    TFile *output = new TFile(TString("../WS_")+TString(dataset_tag)+TString("/dataset_")+TString(dataset_tag)+TString("_")+TString(run_num)+TString(".root"),"recreate");
    TTree *FastJet = new TTree("fastjet", "fastjet");

    std::vector<float> jet_pt, jet_eta, jet_phi, jet_m;
    FastJet->Branch("jet_pt", &jet_pt);
    FastJet->Branch("jet_eta", &jet_eta);
    FastJet->Branch("jet_phi", &jet_phi);
    FastJet->Branch("jet_m", &jet_m);

    std::vector<std::vector<float>> jet_trk_pT, jet_trk_eta, jet_trk_phi, jet_trk_q, jet_trk_d0, jet_trk_z0;
    std::vector<std::vector<int>> jet_trk_origin, jet_trk_pid, jet_trk_fromDown, jet_trk_fromUp, jet_trk_fromBottom;
    FastJet->Branch("jet_trk_pt", &jet_trk_pT);
    FastJet->Branch("jet_trk_eta", &jet_trk_eta);
    FastJet->Branch("jet_trk_phi", &jet_trk_phi);
    FastJet->Branch("jet_trk_q", &jet_trk_q);
    FastJet->Branch("jet_trk_d0", &jet_trk_d0);
    FastJet->Branch("jet_trk_z0", &jet_trk_z0);
    FastJet->Branch("jet_trk_pid", &jet_trk_pid);
    FastJet->Branch("jet_trk_origin", &jet_trk_origin);
    FastJet->Branch("jet_trk_fromDown", &jet_trk_fromDown);
    FastJet->Branch("jet_trk_fromUp", &jet_trk_fromUp);
    FastJet->Branch("jet_trk_fromBottom", &jet_trk_fromBottom);

    std::vector<float> trk_pT, trk_eta, trk_phi, trk_q, trk_d0, trk_z0;
    std::vector<int> trk_origin, trk_pid, trk_fromDown, trk_fromUp, trk_fromBottom;
    FastJet->Branch("trk_pt", &trk_pT);
    FastJet->Branch("trk_eta", &trk_eta);
    FastJet->Branch("trk_phi", &trk_phi);
    FastJet->Branch("trk_q", &trk_q);
    FastJet->Branch("trk_d0", &trk_d0);
    FastJet->Branch("trk_z0", &trk_z0);
    FastJet->Branch("trk_pid", &trk_pid);
    FastJet->Branch("trk_origin", &trk_origin);
    FastJet->Branch("trk_fromDown", &trk_fromDown);
    FastJet->Branch("trk_fromUp", &trk_fromUp);
    FastJet->Branch("trk_fromBottom", &trk_fromBottom);

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
      fastjet::cambridge_algorithm, 1.5, fastjet::E_scheme, fastjet::Best);

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
        //toHepMC.writeNextEvent( pythia );

        // Use depth-first-search to find down daughters
        std::vector<int> fromDown;
        std::vector<int> fromUp;
        std::vector<int> fromBottom;
        int top_idx = find_top_from_event(pythia.event, 6);
        int down_idx = find_down_from_top(pythia.event, top_idx);
        int up_idx = find_up_from_top(pythia.event, top_idx);
        int bottom_idx = find_b_from_top(pythia.event, top_idx);
        fromDown = find_daughters(pythia.event, down_idx);
        fromUp = find_daughters(pythia.event, up_idx);
        fromBottom = find_daughters(pythia.event, bottom_idx);

        // Initialize vector for fastjet clustering and particle index
        std::vector<fastjet::PseudoJet> fastjet_particles;
        int particle_num=0;

        // prepare for filling
        jet_pt.clear(); jet_eta.clear(); jet_phi.clear(); jet_m.clear();
        jet_trk_pT.clear(); jet_trk_eta.clear(); jet_trk_phi.clear(); jet_trk_q.clear(); jet_trk_d0.clear(); jet_trk_z0.clear(); jet_trk_origin.clear(); jet_trk_pid.clear(); jet_trk_fromDown.clear(); jet_trk_fromUp.clear(); jet_trk_fromBottom.clear();
        trk_pT.clear(); trk_eta.clear(); trk_phi.clear(); trk_q.clear(); trk_d0.clear(); trk_z0.clear(); trk_origin.clear(); trk_pid.clear(); trk_fromDown.clear(); trk_fromUp.clear(); trk_fromBottom.clear();


        // Loop through particles in the event
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];

            //std::cout << j << "\t" << p.id() << "\t" << p.status() << "\t" << p.mother1() << "\t" << p.mother2() << "\t" << p.daughter1() << "\t" << p.daughter2() << std::endl;

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
            // Skip soft tracks; Units GeV
            if (p.pT() < 1.0) continue;

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
            trk_fromDown.push_back(fromDown[j]);
            trk_fromUp.push_back(fromUp[j]);
            trk_fromBottom.push_back(fromBottom[j]);
        }

        // Cluster particles using fastjet
        fastjet::ClusterSequence clustSeq(fastjet_particles, jetDefs["fatjet"]);
        auto jets = fastjet::sorted_by_pt( clustSeq.inclusive_jets(pTmin_jet) );

        // Loop through clustered jets
        for (auto jet:jets) {
            jet_pt.push_back(jet.pt()); jet_eta.push_back(jet.eta()); jet_phi.push_back(jet.phi()); jet_m.push_back(jet.m());

            // Temporary vectors with jet constituent info
            std::vector<float> jet_trk_pT_tmp, jet_trk_eta_tmp, jet_trk_phi_tmp, jet_trk_q_tmp, jet_trk_d0_tmp, jet_trk_z0_tmp;
            std::vector<int> jet_trk_origin_tmp, jet_trk_pid_tmp, jet_trk_fromDown_tmp, jet_trk_fromUp_tmp, jet_trk_fromBottom_tmp;

            // Loop through jet constituents
            for (auto trk:jet.constituents()) {
                int idx = trk.user_index();
                auto &p = pythia.event[idx];
                // Skip soft tracks; Units GeV
                if (p.pT() < 1.0) continue;
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
                jet_trk_fromDown_tmp.push_back(fromDown[idx]);
                jet_trk_fromUp_tmp.push_back(fromUp[idx]);
                jet_trk_fromBottom_tmp.push_back(fromBottom[idx]);

            } // End loop through trks

            jet_trk_pT.push_back(jet_trk_pT_tmp);
            jet_trk_eta.push_back(jet_trk_eta_tmp);
            jet_trk_phi.push_back(jet_trk_phi_tmp);
            jet_trk_q.push_back(jet_trk_q_tmp);
            jet_trk_d0.push_back(jet_trk_d0_tmp);
            jet_trk_z0.push_back(jet_trk_z0_tmp);
            jet_trk_origin.push_back(jet_trk_origin_tmp);
            jet_trk_pid.push_back(jet_trk_pid_tmp);
            jet_trk_fromDown.push_back(jet_trk_fromDown_tmp);
            jet_trk_fromUp.push_back(jet_trk_fromUp_tmp);
            jet_trk_fromBottom.push_back(jet_trk_fromBottom_tmp);

        } // End loop through jets

        FastJet->Fill();

    } // End pythia event loop

    output->Write();
    output->Close();

    return 0;
}
