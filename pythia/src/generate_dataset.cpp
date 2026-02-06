#include <iostream>
#include <cmath>
#include <vector>

#include "Pythia8/Pythia.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TRandom3.h"
#include <TRandom.h>
#include "TString.h"
#include "TLorentzVector.h"

#include "include/estimate_ip.h"
#include "include/traverse_history.h"
#include "include/isolated_lepton.h"
#include "include/get_MET.h"
#include "include/remove_particles_from_clustering.h"

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
    TFile *output = new TFile(TString("../WS_")+TString(dataset_tag)+TString("/dataset_selected_")+TString(dataset_tag)+TString("_")+TString(run_num)+TString(".root"),"recreate");
    TTree *fastjet= new TTree("fastjet", "fastjet");
    // Histograms of interesting quantities
    TH1F *h_num_large_jets = new TH1F("h_num_large_jets", "Number of Large-R Jets;N_{jets};Events", 10, 0, 10);
    TH1F *h_num_constituents = new TH1F("h_num_constituents", "Number of Jet Constituents;N_{constituents};Events", 200, 0, 200);
    TH1F *h_num_small_jets = new TH1F("h_num_small_jets", "Number of Small-R Jets;N_{jets};Events", 20, 0, 20);

    float lepton_pT;
    float lepton_eta;
    float lepton_phi;
    int lepton_q;
    float lepton_minDeltaR;
    float nu_MET;
    float nu_phi;
    float probe_jet_pT;
    float probe_jet_eta;
    float probe_jet_phi;
    float probe_jet_mass;
    std::vector<float> probe_jet_constituent_pT;
    std::vector<float> probe_jet_constituent_eta;
    std::vector<float> probe_jet_constituent_phi;
    std::vector<int> probe_jet_constituent_q;
    std::vector<int> probe_jet_constituent_PID;
    std::vector<int> probe_jet_constituent_fromDown;
    std::vector<int> probe_jet_constituent_fromUp;
    std::vector<int> probe_jet_constituent_fromBottom;
    std::vector<float> balance_jets_pT;
    std::vector<float> balance_jets_eta;
    std::vector<float> balance_jets_phi;
    std::vector<int> balance_jets_num_from_b;
    float top_px, top_py, top_pz, top_e;
    float anti_top_px, anti_top_py, anti_top_pz, anti_top_e;
    float down_px, down_py, down_pz, down_e;
    float bottom_px, bottom_py, bottom_pz, bottom_e;
    float top_px_boost_ttCM, top_py_boost_ttCM, top_pz_boost_ttCM, top_e_boost_ttCM;
    float down_px_boost_ttCM, down_py_boost_ttCM, down_pz_boost_ttCM, down_e_boost_ttCM;
    float bottom_px_boost_ttCM, bottom_py_boost_ttCM, bottom_pz_boost_ttCM, bottom_e_boost_ttCM;
    float down_px_boost_tRest, down_py_boost_tRest, down_pz_boost_tRest, down_e_boost_tRest;
    float bottom_px_boost_tRest, bottom_py_boost_tRest, bottom_pz_boost_tRest, bottom_e_boost_tRest;
    float costheta_down;
    float costheta_bottom;
    double beta_x_ttbar_CM, beta_y_ttbar_CM, beta_z_ttbar_CM;
    double beta_x_t_rest, beta_y_t_rest, beta_z_t_rest;

    // Features
    fastjet->Branch("lepton_pT", &lepton_pT);
    fastjet->Branch("lepton_eta", &lepton_eta);
    fastjet->Branch("lepton_phi", &lepton_phi);
    fastjet->Branch("lepton_q", &lepton_q);
    fastjet->Branch("lepton_minDeltaR", &lepton_minDeltaR);
    fastjet->Branch("nu_MET", &nu_MET);
    fastjet->Branch("nu_phi", &nu_phi);
    fastjet->Branch("probe_jet_pT", &probe_jet_pT);
    fastjet->Branch("probe_jet_eta", &probe_jet_eta);
    fastjet->Branch("probe_jet_phi", &probe_jet_phi);
    fastjet->Branch("probe_jet_mass", &probe_jet_mass);
    fastjet->Branch("probe_jet_constituent_pT", &probe_jet_constituent_pT);
    fastjet->Branch("probe_jet_constituent_eta", &probe_jet_constituent_eta);
    fastjet->Branch("probe_jet_constituent_phi", &probe_jet_constituent_phi);
    fastjet->Branch("probe_jet_constituent_q", &probe_jet_constituent_q);
    fastjet->Branch("probe_jet_constituent_PID", &probe_jet_constituent_PID);
    fastjet->Branch("probe_jet_constituent_fromDown", &probe_jet_constituent_fromDown);
    fastjet->Branch("probe_jet_constituent_fromUp", &probe_jet_constituent_fromUp);
    fastjet->Branch("probe_jet_constituent_fromBottom", &probe_jet_constituent_fromBottom);
    fastjet->Branch("balance_jets_pT", &balance_jets_pT);
    fastjet->Branch("balance_jets_eta", &balance_jets_eta);
    fastjet->Branch("balance_jets_phi", &balance_jets_phi);
    fastjet->Branch("balance_jets_num_from_b", &balance_jets_num_from_b);
    // Labels
    fastjet->Branch("top_px_lab", &top_px);
    fastjet->Branch("top_py_lab", &top_py);
    fastjet->Branch("top_pz_lab", &top_pz);
    fastjet->Branch("top_e_lab", &top_e);
    fastjet->Branch("down_px_lab", &down_px);
    fastjet->Branch("down_py_lab", &down_py);
    fastjet->Branch("down_pz_lab", &down_pz);
    fastjet->Branch("down_e_lab", &down_e);
    fastjet->Branch("bottom_px_lab", &bottom_px);
    fastjet->Branch("bottom_py_lab", &bottom_py);
    fastjet->Branch("bottom_pz_lab", &bottom_pz);
    fastjet->Branch("bottom_e_lab", &bottom_e);
    fastjet->Branch("top_px_boost_ttCM", &top_px_boost_ttCM);
    fastjet->Branch("top_py_boost_ttCM", &top_py_boost_ttCM);
    fastjet->Branch("top_pz_boost_ttCM", &top_pz_boost_ttCM);
    fastjet->Branch("top_e_boost_ttCM", &top_e_boost_ttCM);
    fastjet->Branch("down_px_boost_ttCM", &down_px_boost_ttCM);
    fastjet->Branch("down_py_boost_ttCM", &down_py_boost_ttCM);
    fastjet->Branch("down_pz_boost_ttCM", &down_pz_boost_ttCM);
    fastjet->Branch("down_e_boost_ttCM", &down_e_boost_ttCM);
    fastjet->Branch("bottom_px_boost_ttCM", &bottom_px_boost_ttCM);
    fastjet->Branch("bottom_py_boost_ttCM", &bottom_py_boost_ttCM);
    fastjet->Branch("bottom_pz_boost_ttCM", &bottom_pz_boost_ttCM);
    fastjet->Branch("bottom_e_boost_ttCM", &bottom_e_boost_ttCM);
    fastjet->Branch("down_px_boost_tRest", &down_px_boost_tRest);
    fastjet->Branch("down_py_boost_tRest", &down_py_boost_tRest);
    fastjet->Branch("down_pz_boost_tRest", &down_pz_boost_tRest);
    fastjet->Branch("down_e_boost_tRest", &down_e_boost_tRest);
    fastjet->Branch("bottom_px_boost_tRest", &bottom_px_boost_tRest);
    fastjet->Branch("bottom_py_boost_tRest", &bottom_py_boost_tRest);
    fastjet->Branch("bottom_pz_boost_tRest", &bottom_pz_boost_tRest);
    fastjet->Branch("bottom_e_boost_tRest", &bottom_e_boost_tRest);
    fastjet->Branch("costheta_down", &costheta_down);
    fastjet->Branch("costheta_bottom", &costheta_bottom);
    fastjet->Branch("beta_x_ttbar_CM", &beta_x_ttbar_CM);
    fastjet->Branch("beta_y_ttbar_CM", &beta_y_ttbar_CM);
    fastjet->Branch("beta_z_ttbar_CM", &beta_z_ttbar_CM);
    fastjet->Branch("beta_x_t_rest", &beta_x_t_rest);
    fastjet->Branch("beta_y_t_rest", &beta_y_t_rest);
    fastjet->Branch("beta_z_t_rest", &beta_z_t_rest);

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

    // Cutflow counters
    int total_event_counter=0;
    int isolated_lepton_cut=0;
    int missingET_cut=0;
    int fatjet_cut=0;
    int leptonic_top=0;

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
        bottom_px = pythia.event[bottom_idx].px();
        bottom_py = pythia.event[bottom_idx].py();
        bottom_pz = pythia.event[bottom_idx].pz();
        bottom_e = pythia.event[bottom_idx].e();

        /*
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
        */

        TLorentzVector p_t, p_tbar, p_d, p_b;
        p_t = TLorentzVector(top_px, top_py, top_pz, top_e);
        p_tbar = TLorentzVector(anti_top_px, anti_top_py, anti_top_pz, anti_top_e);
        p_d = TLorentzVector(down_px, down_py, down_pz, down_e);
        p_b = TLorentzVector(bottom_px, bottom_py, bottom_pz, bottom_e);

        TVector3 to_ttbar_rest;
        TVector3 to_t_rest;
        TVector3 k_vect;
        TVector3 d_vect;
        TVector3 b_vect;

        // Construct Lorentz boost to t-tbar CM frame
        to_ttbar_rest = -(p_t + p_tbar).BoostVector();
        beta_x_ttbar_CM = to_ttbar_rest.X();
        beta_y_ttbar_CM = to_ttbar_rest.Y();
        beta_z_ttbar_CM = to_ttbar_rest.Z();

        // Boost vectors to t-tbar CM frame
        p_t.Boost(to_ttbar_rest);
        p_tbar.Boost(to_ttbar_rest);
        p_d.Boost(to_ttbar_rest);
        p_b.Boost(to_ttbar_rest);

        // Store kinematics in t-tbar CM frame
        top_px_boost_ttCM = p_t.Px();
        top_py_boost_ttCM = p_t.Py();
        top_pz_boost_ttCM = p_t.Pz();
        top_e_boost_ttCM = p_t.E();
        down_px_boost_ttCM = p_d.Px();
        down_py_boost_ttCM = p_d.Py();
        down_pz_boost_ttCM = p_d.Pz();
        down_e_boost_ttCM = p_d.E();
        bottom_px_boost_ttCM = p_b.Px();
        bottom_py_boost_ttCM = p_b.Py();
        bottom_pz_boost_ttCM = p_b.Pz();
        bottom_e_boost_ttCM = p_b.E();

        // Top quark unit vector in t-tbar CM frame
        k_vect = p_t.Vect().Unit();

        // Construct Lorentz boos to top quark rest frame
        to_t_rest = -p_t.BoostVector();
        beta_x_t_rest = to_t_rest.X();
        beta_y_t_rest = to_t_rest.Y();
        beta_z_t_rest = to_t_rest.Z();

        // Boost quarks to top quark rest frame
        p_d.Boost(to_t_rest);
        p_b.Boost(to_t_rest);

        // Store labels in top rest frame
        down_px_boost_tRest = p_d.Px();
        down_py_boost_tRest = p_d.Py();
        down_pz_boost_tRest = p_d.Pz();
        down_e_boost_tRest = p_d.E();
        bottom_px_boost_tRest = p_b.Px();
        bottom_py_boost_tRest = p_b.Py();
        bottom_pz_boost_tRest = p_b.Pz();
        bottom_e_boost_tRest = p_b.E();

        // Calc unit vector in top rest frame
        d_vect = p_d.Vect().Unit();
        b_vect = p_b.Vect().Unit();

        // Calc dot product for cos theta
        costheta_down = k_vect.Dot(d_vect);
        costheta_bottom = k_vect.Dot(b_vect);

        // START ANALYSIS SELECTIONS

        // Initialize vector for fastjet clustering and particle index
        std::vector<fastjet::PseudoJet> fastjet_particles;

        // Initialize aux info
        std::vector<int> p_PID;
        std::vector<int> p_q;
        std::vector<int> p_fromDown;
        std::vector<int> p_fromUp;
        std::vector<int> p_fromBottom;
        std::vector<int> p_fromLepton;
        std::vector<int> p_fromNu;

        int particle_num=0;

        // Clear output vectors
        probe_jet_constituent_pT.clear(); probe_jet_constituent_eta.clear(); probe_jet_constituent_phi.clear(); probe_jet_constituent_q.clear(); probe_jet_constituent_PID.clear();
        probe_jet_constituent_fromDown.clear(); probe_jet_constituent_fromUp.clear(); probe_jet_constituent_fromBottom.clear();
        balance_jets_pT.clear(); balance_jets_eta.clear(); balance_jets_phi.clear(); balance_jets_num_from_b.clear();

        // Loop through particles in the event
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];

            //std::cout << j << "\t" << p.id() << "\t" << p.status() << "\t" << p.mother1() << "\t" << p.mother2() << "\t" << p.daughter1() << "\t" << p.daughter2() << std::endl;

            // Do not consider intermediate particles for clustering
            if (not p.isFinal()) continue;

            // Convert particles to PseduoJet object, set the user idx, and append to the list of fastjet particles
            fastjet::PseudoJet fj(p.px(), p.py(), p.pz(), p.e());
            fj.set_user_index(particle_num++); // 0 based
            fastjet_particles.push_back(fj);

            p_PID.push_back(p.id());
            p_q.push_back(p.charge());

            p_fromDown.push_back(fromDown[j]);
            p_fromUp.push_back(fromUp[j]);
            p_fromBottom.push_back(fromBottom[j]);
            p_fromLepton.push_back(fromLepton[j]);
            p_fromNu.push_back(fromNu[j]);
        }

        // Find the isolated lepton
        int isolated_lepton_idx = isolated_lepton(fastjet_particles, &p_fromLepton, &p_PID, &lepton_minDeltaR);

        // Get the lepton kinematics
        fastjet::PseudoJet lepton = fastjet_particles[isolated_lepton_idx];
        lepton_pT  = lepton.pt();
        lepton_eta = lepton.eta();
        lepton_phi = lepton.phi();
        lepton_q   = p_q[isolated_lepton_idx];

        // Skip the event if lepton is not isolated, less than 30GeV or |eta|>3
        if (lepton_minDeltaR < 0.1 || lepton_pT < 30 || std::abs(lepton_eta)>3){
            isolated_lepton_cut++;
            continue;
        }

        // Get the neutrino
        int MET_idx = get_MET(fastjet_particles, &p_fromNu, &p_PID);

        // Store neutrino kinematics
        nu_MET = fastjet_particles[MET_idx].Et();
        nu_phi = fastjet_particles[MET_idx].phi();

        // Skip event if the neutrino is less than 30GeV
        if (nu_MET < 30){
            missingET_cut++;
            continue;
        }

        // Do not consider neutrinos in clustering
        std::vector<int> remove_IDs;
        for (int j=0; j<p_PID.size(); j++){
            if (std::abs(p_PID[j])==12 || std::abs(p_PID[j])==14 || std::abs(p_PID[j])==16) remove_IDs.push_back(j);
        }

        // Remove the lepton and neutrino from clustering!
        remove_IDs.push_back(isolated_lepton_idx);
        std::vector<fastjet::PseudoJet> particles_no_lepton = remove_particles_from_clustering(fastjet_particles, remove_IDs);

        // Cluster particles and pick up hardest largeR jet
        float R_large = 1.0;
        float pTmin_jet_large = 250; // GeV
        fastjet::JetDefinition jetDef_large = fastjet::JetDefinition(fastjet::cambridge_algorithm, R_large, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_large(particles_no_lepton, jetDef_large);
        auto jets_large = fastjet::sorted_by_pt( clustSeq_large.inclusive_jets(pTmin_jet_large) );

        h_num_large_jets->Fill(jets_large.size());

        // Skip event if no jets are clustered
        if (jets_large.size()==0){
            fatjet_cut++;
            continue;
        }

        // Get kinematics of the hardest jet
        fastjet::PseudoJet hardest_jet = jets_large[0];
        probe_jet_pT  = hardest_jet.pt();
        probe_jet_eta = hardest_jet.eta();
        probe_jet_phi = hardest_jet.phi();
        probe_jet_mass= hardest_jet.m();

        // Skip event if jet has |eta|>3
        if (std::abs(probe_jet_eta)>3){
            fatjet_cut++;
            continue;
        }

        // Store jet constituents
        std::vector<int> hardest_jet_constituents;
        for (auto trk:hardest_jet.constituents()){
            if (trk.pt() > 0.4 and std::abs(trk.eta()) < 4.5){
               hardest_jet_constituents.push_back(trk.user_index());
               probe_jet_constituent_pT.push_back(trk.pt());
               probe_jet_constituent_eta.push_back(trk.eta());
               probe_jet_constituent_phi.push_back(trk.phi());
               probe_jet_constituent_q.push_back(p_q[trk.user_index()]);
               probe_jet_constituent_PID.push_back(p_PID[trk.user_index()]);
               probe_jet_constituent_fromDown.push_back(p_fromDown[trk.user_index()]);
               probe_jet_constituent_fromUp.push_back(p_fromUp[trk.user_index()]);
               probe_jet_constituent_fromBottom.push_back(p_fromBottom[trk.user_index()]);
            }
        }
        h_num_constituents->Fill(hardest_jet.constituents().size());

        // Remove the constituents from clustering
        std::vector<fastjet::PseudoJet> particles_no_fatjet= remove_particles_from_clustering(particles_no_lepton, hardest_jet_constituents);

        // Cluster small R jets
        float R_small = 0.4;
        float pTmin_jet_small = 10; // GeV
        fastjet::JetDefinition jetDef_small = fastjet::JetDefinition(fastjet::antikt_algorithm, R_small, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_small(particles_no_fatjet, jetDef_small);
        auto jets_small = fastjet::sorted_by_pt( clustSeq_small.inclusive_jets(pTmin_jet_small) );
        h_num_small_jets->Fill(jets_small.size());

        // Ensure there is a small R jet on leptonic side
        int selected_small_jet=0;
        float lep_jet_dR;
        for (auto jet:jets_small){
            lep_jet_dR = lepton.delta_R(jet);
            if (lep_jet_dR > 0.3 && jet.pt() > 30 && std::abs(jet.eta()) < 3){
                selected_small_jet=1;
            }
            int b_counter=0;
            for (auto trk:jet.constituents()){
                if (p_fromBottom[trk.user_index()]==1){b_counter++;}
            }
            balance_jets_num_from_b.push_back(b_counter);
        }
        if (selected_small_jet==0){
            leptonic_top++;
            continue;
        }

        // Store smallR jet kinematics
        for (auto jet:jets_small){
            balance_jets_pT.push_back(jet.pt());
            balance_jets_eta.push_back(jet.eta());
            balance_jets_phi.push_back(jet.phi());
        }

        // Fill ROOT file
        fastjet->Fill();
    }

    // Print out cutflow
    std::cout << "Total Events: " << total_event_counter << std::endl;
    std::cout << "Isolated Lepton Cut: " << isolated_lepton_cut << std::endl;
    std::cout << "MissingET Cut: " << missingET_cut << std::endl;
    std::cout << "FatJet Cut: " << fatjet_cut << std::endl;
    std::cout << "SmallR Jet Cut: " << leptonic_top << std::endl;
    std::cout << "Remaining Events: " << total_event_counter - isolated_lepton_cut - missingET_cut - fatjet_cut - leptonic_top << std::endl;

    // Write out ROOT file
    output->Write();
    output->Close();

    return 0;
}
