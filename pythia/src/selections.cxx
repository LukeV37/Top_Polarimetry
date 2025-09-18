#include <iostream>
#include <cmath>
#include <vector>

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include <TRandom.h>
#include "TString.h"

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

    // Open file and extract training tree
    TString inputFile = TString("../WS_")+TString(dataset_tag)+TString("/dataset_")+TString(dataset_tag)+TString("_")+TString(run_num)+TString(".root");
    TFile *input_file = new TFile(inputFile, "READ");
    TTree *pythia = (TTree*)input_file->Get("pythia");

    // Initialize variables, declare new branch, set needed old branches
    std::vector<float> *px = 0;
    std::vector<float> *py = 0;
    std::vector<float> *pz = 0;
    std::vector<float> *e = 0;
    std::vector<int> *q = 0;
    std::vector<int> *pid = 0;
    std::vector<int> *fromDown = 0;
    std::vector<int> *fromUp = 0;
    std::vector<int> *fromBottom = 0;
    std::vector<int> *fromLepton = 0;
    std::vector<int> *fromNu = 0;
    std::vector<int> *fromAntiBottom = 0;
    float top_px, top_py, top_pz, top_e;
    float top_px_boosted, top_py_boosted, top_pz_boosted, top_e_boosted;
    float down_px_boosted, down_py_boosted, down_pz_boosted, down_e_boosted;
    float costheta;
    pythia->SetBranchAddress("p_px", &px);
    pythia->SetBranchAddress("p_py", &py);
    pythia->SetBranchAddress("p_pz", &pz);
    pythia->SetBranchAddress("p_e", &e);
    pythia->SetBranchAddress("p_q", &q);
    pythia->SetBranchAddress("p_pid", &pid);
    pythia->SetBranchAddress("p_fromDown", &fromDown);
    pythia->SetBranchAddress("p_fromUp", &fromUp);
    pythia->SetBranchAddress("p_fromBottom", &fromBottom);
    pythia->SetBranchAddress("p_fromLepton", &fromLepton);
    pythia->SetBranchAddress("p_fromNu", &fromNu);
    pythia->SetBranchAddress("p_fromAntiBottom", &fromAntiBottom);
    pythia->SetBranchAddress("top_px", &top_px);
    pythia->SetBranchAddress("top_py", &top_py);
    pythia->SetBranchAddress("top_pz", &top_pz);
    pythia->SetBranchAddress("top_e", &top_e);
    pythia->SetBranchAddress("top_px_boosted", &top_px_boosted);
    pythia->SetBranchAddress("top_py_boosted", &top_py_boosted);
    pythia->SetBranchAddress("top_pz_boosted", &top_pz_boosted);
    pythia->SetBranchAddress("top_e_boosted", &top_e_boosted);
    pythia->SetBranchAddress("down_px_boosted", &down_px_boosted);
    pythia->SetBranchAddress("down_py_boosted", &down_py_boosted);
    pythia->SetBranchAddress("down_pz_boosted", &down_pz_boosted);
    pythia->SetBranchAddress("down_e_boosted", &down_e_boosted);
    pythia->SetBranchAddress("costheta", &costheta);

    // Initialize output ROOT file, TTree, and Branches
    TFile *output = new TFile(TString("../WS_")+TString(dataset_tag)+TString("/dataset_selected_")+TString(dataset_tag)+TString("_")+TString(run_num)+TString(".root"),"recreate");
    TTree *fastjet= new TTree("fastjet", "fastjet");

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
    float truth_top_pT, truth_top_eta, truth_top_phi;

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
    fastjet->Branch("truth_top_pT", &truth_top_pT);
    fastjet->Branch("truth_top_eta", &truth_top_eta);
    fastjet->Branch("truth_top_phi", &truth_top_phi);
    fastjet->Branch("truth_top_px_boosted", &top_px_boosted);
    fastjet->Branch("truth_top_py_boosted", &top_py_boosted);
    fastjet->Branch("truth_top_pz_boosted", &top_pz_boosted);
    fastjet->Branch("truth_top_e_boosted", &top_e_boosted);
    fastjet->Branch("truth_down_px_boosted", &down_px_boosted);
    fastjet->Branch("truth_down_py_boosted", &down_py_boosted);
    fastjet->Branch("truth_down_pz_boosted", &down_pz_boosted);
    fastjet->Branch("truth_down_e_boosted", &down_e_boosted);
    fastjet->Branch("costheta", &costheta);

    int total_event_counter=0;
    int isolated_lepton_cut=0;
    int missingET_cut=0;
    int fatjet_cut=0;
    int leptonic_top=0;

    // Loop over events
    int nEntries = pythia->GetEntries();
    for (int event=0; event<nEntries; event++){
        total_event_counter++;
        pythia->GetEntry(event);

        // Initialize vector of fastjet objects
        std::vector<fastjet::PseudoJet> fastjet_particles;

        // Clear output vectors
        probe_jet_constituent_pT.clear(); probe_jet_constituent_eta.clear(); probe_jet_constituent_phi.clear(); probe_jet_constituent_q.clear(); probe_jet_constituent_PID.clear();
        probe_jet_constituent_fromDown.clear(); probe_jet_constituent_fromUp.clear(); probe_jet_constituent_fromBottom.clear();
        balance_jets_pT.clear(); balance_jets_eta.clear(); balance_jets_phi.clear();
        
        // Store particles in fastjet::PseudoJet objects and set the index
        int num_particles = px->size();
        for (int particle=0; particle<num_particles; particle++){
            fastjet::PseudoJet fj(px->at(particle), py->at(particle), pz->at(particle), e->at(particle));
            fj.set_user_index(particle);
            fastjet_particles.push_back(fj);
        }

        // Find the isolated lepton
        int isolated_lepton_idx = isolated_lepton(fastjet_particles, fromLepton, pid, &lepton_minDeltaR);

        // Get the lepton kinematics
        fastjet::PseudoJet lepton = fastjet_particles[isolated_lepton_idx];
        lepton_pT  = lepton.pt();
        lepton_eta = lepton.eta();
        lepton_phi = lepton.phi();
        lepton_q   = q->at(isolated_lepton_idx);

        // Skip the event if lepton is not isolated, less than 30GeV or |eta|>3
        if (lepton_minDeltaR < 0.1 || lepton_pT < 30 || std::abs(lepton_eta)>3){
            isolated_lepton_cut++;
            continue;
        }

        // Get the neutrino
        int nu_idx = get_MET(fastjet_particles, fromNu, pid);

        // Store neutrino kinematics
        nu_MET = fastjet_particles[nu_idx].Et();
        nu_phi = fastjet_particles[nu_idx].phi();

        // Skip event if the neutrino is less than 30GeV
        if (nu_MET < 30){
            missingET_cut++;
            continue;
        }

        // Remove the lepton and neutrino from clustering!
        std::vector<int> lepton_nu_idx;
        lepton_nu_idx.push_back(isolated_lepton_idx);
        lepton_nu_idx.push_back(nu_idx);
        std::vector<fastjet::PseudoJet> particles_no_lepton = remove_particles_from_clustering(fastjet_particles, lepton_nu_idx);

        // Cluster particles and pick up hardest largeR jet
        float R_large = 1.5;
        float pTmin_jet_large = 250; // GeV
        fastjet::JetDefinition jetDef_large = fastjet::JetDefinition(fastjet::cambridge_algorithm, R_large, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_large(particles_no_lepton, jetDef_large);
        auto jets_large = fastjet::sorted_by_pt( clustSeq_large.inclusive_jets(pTmin_jet_large) );

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
               probe_jet_constituent_q.push_back(q->at(trk.user_index()));
               probe_jet_constituent_PID.push_back(pid->at(trk.user_index()));
               probe_jet_constituent_fromDown.push_back(fromDown->at(trk.user_index()));
               probe_jet_constituent_fromUp.push_back(fromUp->at(trk.user_index()));
               probe_jet_constituent_fromBottom.push_back(fromBottom->at(trk.user_index()));
            }
        }

        // Remove the constituents from clustering
        std::vector<fastjet::PseudoJet> particles_no_fatjet= remove_particles_from_clustering(particles_no_lepton, hardest_jet_constituents);

        // Cluster small R jets
        float R_small = 0.4;
        float pTmin_jet_small = 10; // GeV
        fastjet::JetDefinition jetDef_small = fastjet::JetDefinition(fastjet::antikt_algorithm, R_small, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_small(particles_no_fatjet, jetDef_small);
        auto jets_small = fastjet::sorted_by_pt( clustSeq_small.inclusive_jets(pTmin_jet_small) );

        // Ensure there is a small R jet on leptonic side
        int selected_small_jet=0;
        float lep_jet_dR;
        for (auto jet:jets_small){
            lep_jet_dR = lepton.delta_R(jet);
            if (lep_jet_dR > 0.3 && jet.pt() > 30 && jet.eta() < 3){
                selected_small_jet=1;
            }
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

        // Store truth top kinematics
        fastjet::PseudoJet truth_top(top_px, top_py, top_pz, top_e);
        truth_top_pT = truth_top.pt();
        truth_top_eta = truth_top.eta();
        truth_top_phi = truth_top.phi();

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
