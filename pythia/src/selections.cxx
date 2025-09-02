#include <iostream>
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

    // Initialize output ROOT file, TTree, and Branches
    TFile *output = new TFile(TString("../WS_")+TString(dataset_tag)+TString("/dataset_selected_")+TString(dataset_tag)+TString("_")+TString(run_num)+TString(".root"),"recreate");
    TTree *fastjet= new TTree("fastjet", "fastjet");

    float lepton_pT;
    float lepton_eta;
    float lepton_phi;
    int lepton_q;
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
    std::vector<float> balance_jets_pT;
    std::vector<float> balance_jets_eta;
    std::vector<float> balance_jets_phi;

    fastjet->Branch("lepton_pT", &lepton_pT);
    fastjet->Branch("lepton_eta", &lepton_eta);
    fastjet->Branch("lepton_phi", &lepton_phi);
    fastjet->Branch("lepton_q", &lepton_q);
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
    fastjet->Branch("balance_jets_pT", &balance_jets_pT);
    fastjet->Branch("balance_jets_eta", &balance_jets_eta);
    fastjet->Branch("balance_jets_phi", &balance_jets_phi);

    // Loop over entries and determine the truth label
    int nEntries = pythia->GetEntries();
    for (int event=0; event<nEntries; event++){
        pythia->GetEntry(event);

        // Initialize vector of fastjet objects
        std::vector<fastjet::PseudoJet> fastjet_particles;

        // Clear output vectors
        probe_jet_constituent_pT.clear(); probe_jet_constituent_eta.clear(); probe_jet_constituent_phi.clear(); probe_jet_constituent_q.clear(); probe_jet_constituent_PID.clear();
        balance_jets_pT.clear(); balance_jets_eta.clear(); balance_jets_phi.clear();
        
        int num_particles = px->size();
        for (int particle=0; particle<num_particles; particle++){
            fastjet::PseudoJet fj(px->at(particle), py->at(particle), pz->at(particle), e->at(particle));
            fj.set_user_index(particle);
            fastjet_particles.push_back(fj);
        }
        int isolated_lepton_idx = isolated_lepton(fastjet_particles, fromLepton, pid);
        std::vector<int> lepton_nu_idx;
        lepton_nu_idx.push_back(isolated_lepton_idx);

        lepton_pT  = fastjet_particles[isolated_lepton_idx].pt();
        lepton_eta = fastjet_particles[isolated_lepton_idx].eta();
        lepton_phi = fastjet_particles[isolated_lepton_idx].phi();
        lepton_q   = q->at(isolated_lepton_idx);

        int nu_idx = get_MET(fastjet_particles, fromNu, pid);
        lepton_nu_idx.push_back(nu_idx);

        nu_MET = fastjet_particles[nu_idx].Et();
        nu_phi = fastjet_particles[nu_idx].phi();

        std::vector<fastjet::PseudoJet> particles_no_lepton = remove_particles_from_clustering(fastjet_particles, lepton_nu_idx);
        //std::cout << fastjet_particles.size() << " " << particles_no_lepton.size() << std::endl;

        // Cluster particles and pick up hardest largeR jet
        float R_large = 1.5;
        float pTmin_jet_large = 250; // GeV
        fastjet::JetDefinition jetDef_large = fastjet::JetDefinition(fastjet::cambridge_algorithm, R_large, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_large(particles_no_lepton, jetDef_large);
        auto jets_large = fastjet::sorted_by_pt( clustSeq_large.inclusive_jets(pTmin_jet_large) );

        if (jets_large.size()==0) continue;
        fastjet::PseudoJet hardest_jet = jets_large[0];

        probe_jet_pT  = hardest_jet.pt();
        probe_jet_eta = hardest_jet.eta();
        probe_jet_phi = hardest_jet.phi();

        std::vector<int> hardest_jet_constituents;
        for (auto trk:hardest_jet.constituents()){
           hardest_jet_constituents.push_back(trk.user_index());
           probe_jet_constituent_pT.push_back(trk.pt());
           probe_jet_constituent_eta.push_back(trk.eta());
           probe_jet_constituent_phi.push_back(trk.phi());
        }

        std::vector<fastjet::PseudoJet> particles_no_fatjet= remove_particles_from_clustering(particles_no_lepton, hardest_jet_constituents);

        float R_small = 0.4;
        float pTmin_jet_small = 10; // GeV
        fastjet::JetDefinition jetDef_small = fastjet::JetDefinition(fastjet::cambridge_algorithm, R_small, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_small(particles_no_fatjet, jetDef_small);
        auto jets_small = fastjet::sorted_by_pt( clustSeq_small.inclusive_jets(pTmin_jet_small) );

        for (auto jet:jets_small){
            balance_jets_pT.push_back(jet.pt());
            balance_jets_eta.push_back(jet.eta());
            balance_jets_phi.push_back(jet.phi());
        }

        fastjet->Fill();
    }
    output->Write();
    output->Close();
    return 0;
}
