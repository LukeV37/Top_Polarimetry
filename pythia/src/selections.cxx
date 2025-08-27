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
#include "include/remove_particles_from_clustering.h"

int main(int argc, char *argv[])
{
    //std::string inputFile = std::string("dataset_U_10k_test_0.root");
    TString inputFile = "../WS_U_10k_test/dataset_U_10k_test_0.root";

    // Open file and extract training tree
    TFile *input_file = new TFile(inputFile, "READ");
    TTree *pythia = (TTree*)input_file->Get("pythia");

    // Initialize variables, declare new branch, set needed old branches
    std::vector<float> *px = 0;
    std::vector<float> *py = 0;
    std::vector<float> *pz = 0;
    std::vector<float> *e = 0;
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
    pythia->SetBranchAddress("p_pid", &pid);
    pythia->SetBranchAddress("p_fromDown", &fromDown);
    pythia->SetBranchAddress("p_fromUp", &fromUp);
    pythia->SetBranchAddress("p_fromBottom", &fromBottom);
    pythia->SetBranchAddress("p_fromLepton", &fromLepton);
    pythia->SetBranchAddress("p_fromNu", &fromNu);
    pythia->SetBranchAddress("p_fromAntiBottom", &fromAntiBottom);

    // Loop over entries and determine the truth label
    int nEntries = pythia->GetEntries();
    for (int event=0; event<nEntries; event++){
        pythia->GetEntry(event);

        // Initialize vector of fastjet objects
        std::vector<fastjet::PseudoJet> fastjet_particles;
        
        int num_particles = px->size();
        for (int particle=0; particle<num_particles; particle++){
            fastjet::PseudoJet fj(px->at(particle), py->at(particle), pz->at(particle), e->at(particle));
            fj.set_user_index(particle);
            fastjet_particles.push_back(fj);
        }
        int isolated_lepton_idx = isolated_lepton(fastjet_particles, fromLepton, pid);
        std::vector<int> lepton_idx;
        lepton_idx.push_back(isolated_lepton_idx);

        std::vector<fastjet::PseudoJet> particles_no_lepton = remove_particles_from_clustering(fastjet_particles, lepton_idx);
        //std::cout << fastjet_particles.size() << " " << particles_no_lepton.size() << std::endl;

        // Cluster particles and pick up hardest largeR jet
        float R_large = 1.5;
        float pTmin_jet_large = 250; // GeV
        fastjet::JetDefinition jetDef_large = fastjet::JetDefinition(fastjet::cambridge_algorithm, R_large, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_large(particles_no_lepton, jetDef_large);
        auto jets_large = fastjet::sorted_by_pt( clustSeq_large.inclusive_jets(pTmin_jet_large) );

        if (jets_large.size()==0) continue;
        fastjet::PseudoJet hardest_jet = jets_large[0];
        std::vector<int> hardest_jet_constituents;

        for (auto trk:hardest_jet.constituents()){
           hardest_jet_constituents.push_back(trk.user_index());
        }
        std::vector<fastjet::PseudoJet> particles_no_fatjet= remove_particles_from_clustering(particles_no_lepton, hardest_jet_constituents);
        std::cout << fastjet_particles.size() << " " << particles_no_lepton.size() << " " << particles_no_fatjet.size() << std::endl;

        float R_small = 0.4;
        float pTmin_jet_small = 10; // GeV
        fastjet::JetDefinition jetDef_small = fastjet::JetDefinition(fastjet::cambridge_algorithm, R_small, fastjet::E_scheme, fastjet::Best);
        fastjet::ClusterSequence clustSeq_small(particles_no_fatjet, jetDef_small);
        auto jets_small = fastjet::sorted_by_pt( clustSeq_small.inclusive_jets(pTmin_jet_small) );

        std::cout << "Num Small Jets: " << jets_small.size() << std::endl;

    }

    return 0;
}
