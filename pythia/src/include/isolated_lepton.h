int isolated_lepton(std::vector<fastjet::PseudoJet> fj, std::vector<int> *fromLepton, std::vector<int> *pid)
{
    int lepton_idx=-1;
    for (int i=0; i<fromLepton->size(); i++){
        if (fromLepton->at(i)==1){
            if (pid->at(i)==11 || pid->at(i)==13){
                lepton_idx=i;
            }
        }
    }

    int isIsolated = 1;
    float Isolation_deltaR = 0.1;
    fastjet::PseudoJet lepton = fj[lepton_idx];
    for (int i=0; i<fj.size(); i++){
        if (i==lepton_idx) continue;
        if (lepton.delta_R(fj[i]) < Isolation_deltaR) isIsolated=0;
    }

    std::cout << "Lepton is Isolated: " << isIsolated << std::endl;

    return lepton_idx;
}
