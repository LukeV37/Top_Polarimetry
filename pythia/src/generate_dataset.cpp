#include <iostream>
#include <vector>

#include "Pythia8/Pythia.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include <TRandom.h>
#include "TString.h"
#include "TLorentzVector.h"

#include "include/estimate_ip.h"
#include "include/traverse_history.h"
#include "include/calc_labels.h"

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
    std::cout << inputFile << std::endl;

    // Initialize Pythia Settings
    Pythia8::Pythia pythia;
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = "+inputFile);
    pythia.readString("Next:numberCount = 1000");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    // Initialize output ROOT file, TTree, and Branches
    TFile *output = new TFile(TString("../WS_")+TString(dataset_tag)+TString("/dataset_")+TString(dataset_tag)+TString("_")+TString(run_num)+TString(".root"),"recreate");
    TTree *Pythia = new TTree("pythia", "pythia");

    std::vector<float> p_pT, p_eta, p_phi, p_q, p_d0, p_z0, p_px, p_py, p_pz, p_e;
    std::vector<int> p_pid, p_fromDown, p_fromUp, p_fromBottom, p_fromLepton, p_fromNu, p_fromAntiBottom;
    Pythia->Branch("p_px", &p_px);
    Pythia->Branch("p_py", &p_py);
    Pythia->Branch("p_pz", &p_pz);
    Pythia->Branch("p_e", &p_e);
    Pythia->Branch("p_pt", &p_pT);
    Pythia->Branch("p_eta", &p_eta);
    Pythia->Branch("p_phi", &p_phi);
    Pythia->Branch("p_q", &p_q);
    Pythia->Branch("p_d0", &p_d0);
    Pythia->Branch("p_z0", &p_z0);
    Pythia->Branch("p_pid", &p_pid);
    Pythia->Branch("p_fromDown", &p_fromDown);
    Pythia->Branch("p_fromUp", &p_fromUp);
    Pythia->Branch("p_fromBottom", &p_fromBottom);
    Pythia->Branch("p_fromLepton", &p_fromLepton);
    Pythia->Branch("p_fromNu", &p_fromNu);
    Pythia->Branch("p_fromAntiBottom", &p_fromAntiBottom);

    float top_px, top_py, top_pz, top_e;
    float anti_top_px, anti_top_py, anti_top_pz, anti_top_e;
    float down_px, down_py, down_pz, down_e;
    float top_px_boosted, top_py_boosted, top_pz_boosted, top_e_boosted;
    float down_px_boosted, down_py_boosted, down_pz_boosted, down_e_boosted;
    float costheta;
    TLorentzVector p_t, p_tbar, p_d;
    Pythia->Branch("top_px", &top_px);
    Pythia->Branch("top_py", &top_py);
    Pythia->Branch("top_pz", &top_pz);
    Pythia->Branch("top_e", &top_e);
    Pythia->Branch("anti_top_px", &anti_top_px);
    Pythia->Branch("anti_top_py", &anti_top_py);
    Pythia->Branch("anti_top_pz", &anti_top_pz);
    Pythia->Branch("anti_top_e", &anti_top_e);
    Pythia->Branch("down_px", &down_px);
    Pythia->Branch("down_py", &down_py);
    Pythia->Branch("down_pz", &down_pz);
    Pythia->Branch("down_e", &down_e);
    Pythia->Branch("top_px_boosted", &top_px_boosted);
    Pythia->Branch("top_py_boosted", &top_py_boosted);
    Pythia->Branch("top_pz_boosted", &top_pz_boosted);
    Pythia->Branch("top_e_boosted", &top_e_boosted);
    Pythia->Branch("down_px_boosted", &down_px_boosted);
    Pythia->Branch("down_py_boosted", &down_py_boosted);
    Pythia->Branch("down_pz_boosted", &down_pz_boosted);
    Pythia->Branch("down_e_boosted", &down_e_boosted);
    Pythia->Branch("costheta", &costheta);

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

    int event_no = 0;

    // Begin Event Loop; generate until none left in input file
    while (iAbort < nAbort) {

        // Generate events, and check whether generation failed.
        if (!pythia.next()) {
          // If failure because reached end of file then exit event loop.
          if (pythia.info.atEndOfFile()) break;
          ++iAbort;
          continue;
        }

        // Use depth-first-search to find down daughters
        std::vector<int> fromDown;
        std::vector<int> fromUp;
        std::vector<int> fromBottom;
        std::vector<int> fromLepton;
        std::vector<int> fromNu;
        std::vector<int> fromAntiBottom;

        int top_idx = find_top_from_event(pythia.event, 6);
        int down_idx = find_down_from_top(pythia.event, top_idx);
        int up_idx = find_up_from_top(pythia.event, top_idx);
        int bottom_idx = find_b_from_top(pythia.event, top_idx);
        fromDown = find_daughters(pythia.event, down_idx);
        fromUp = find_daughters(pythia.event, up_idx);
        fromBottom = find_daughters(pythia.event, bottom_idx);

        int anti_top_idx = find_top_from_event(pythia.event, -6);
        int lepton_idx = find_lep_from_top(pythia.event, anti_top_idx);
        int nu_idx = find_nu_from_top(pythia.event, anti_top_idx);
        int anti_bottom_idx = find_b_from_top(pythia.event, anti_top_idx);
        fromLepton = find_daughters(pythia.event, lepton_idx);
        fromNu = find_daughters(pythia.event, nu_idx);
        fromAntiBottom = find_daughters(pythia.event, anti_bottom_idx);

        top_px = pythia.event[top_idx].px();
        top_py = pythia.event[top_idx].py();
        top_pz = pythia.event[top_idx].pz();
        top_e = pythia.event[top_idx].e();
        anti_top_px = pythia.event[anti_top_idx].px();
        anti_top_py = pythia.event[anti_top_idx].py();
        anti_top_pz = pythia.event[anti_top_idx].pz();
        anti_top_e = pythia.event[anti_top_idx].e();
        down_px = pythia.event[down_idx].px();
        down_py = pythia.event[down_idx].py();
        down_pz = pythia.event[down_idx].pz();
        down_e = pythia.event[down_idx].e();

        p_t = TLorentzVector(top_px, top_py, top_pz, top_e);
        p_tbar = TLorentzVector(anti_top_px, anti_top_py, anti_top_pz, anti_top_e);
        p_d = TLorentzVector(down_px, down_py, down_pz, down_e);

        costheta = calc_costheta(p_t, p_tbar, p_d, &top_px_boosted, &top_py_boosted, &top_pz_boosted, &top_e_boosted, &down_px_boosted, &down_py_boosted, &down_pz_boosted, &down_e_boosted);

        if (event_no==0){
            std::cout << "top_idx: " << top_idx << std::endl;
            std::cout << "down_idx: " << down_idx << std::endl;
            std::cout << "up_idx: " << up_idx << std::endl;
            std::cout << "bottom_idx: " << bottom_idx << std::endl;
            std::cout << "anti_top_idx: " << anti_top_idx << std::endl;
            std::cout << "lepton_idx: " << lepton_idx << std::endl;
            std::cout << "nu_idx: " << nu_idx << std::endl;
            std::cout << "anti_bottom_idx: " << anti_bottom_idx << std::endl;
            event_no++;
        }

        // Initialize vector for fastjet clustering and particle index
        std::vector<fastjet::PseudoJet> fastjet_particles;
        int particle_num=0;

        // prepare for filling
        p_px.clear(); p_py.clear(); p_pz.clear(); p_e.clear(); p_pT.clear(); p_eta.clear(); p_phi.clear(); p_q.clear(); p_d0.clear(); p_z0.clear();
        p_pid.clear(); p_fromDown.clear(); p_fromUp.clear(); p_fromBottom.clear(); p_fromLepton.clear(); p_fromNu.clear(); p_fromAntiBottom.clear();


        // Loop through particles in the event
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];

            //std::cout << j << "\t" << p.id() << "\t" << p.status() << "\t" << p.mother1() << "\t" << p.mother2() << "\t" << p.daughter1() << "\t" << p.daughter2() << std::endl;

            // Do not consider intermediate particles for clustering
            if (not p.isFinal()) continue;
            // Do not consider neutrinos in clustering
            //if (std::abs(p.id())==12 || std::abs(p.id())==14 || std::abs(p.id())==16) continue;

            // Convert particles to PseduoJet object, set the user idx, and append to the list of fastjet particles
            fastjet::PseudoJet fj(p.px(), p.py(), p.pz(), p.e());
            fj.set_user_index(particle_num++); // 0 based
            fastjet_particles.push_back(fj);

            // Fill trk vector with all fastjet candidates
            p_px.push_back(p.px());
            p_py.push_back(p.py());
            p_pz.push_back(p.pz());
            p_e.push_back(p.e());
            p_pT.push_back(p.pT());
            p_eta.push_back(p.eta());
            p_phi.push_back(p.phi());
            p_q.push_back(p.charge());
            double d0,z0; find_ip(p.pT(),p.eta(),p.phi(),p.xProd(),p.yProd(),p.zProd(),d0,z0);
            p_d0.push_back(d0);
            p_z0.push_back(z0);
            p_pid.push_back(p.id());
            p_fromDown.push_back(fromDown[j]);
            p_fromUp.push_back(fromUp[j]);
            p_fromBottom.push_back(fromBottom[j]);
            p_fromLepton.push_back(fromLepton[j]);
            p_fromNu.push_back(fromNu[j]);
            p_fromAntiBottom.push_back(fromAntiBottom[j]);
        }

        Pythia->Fill();

    } // End pythia event loop

    output->Write();
    output->Close();

    return 0;
}
