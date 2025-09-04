int isolated_lepton(std::vector<fastjet::PseudoJet> fj, std::vector<int> *fromLepton, std::vector<int> *pid, float *minDeltaR)
{
    int lepton_idx=-1;
    for (int i=0; i<fromLepton->size(); i++){
        if (fromLepton->at(i)==1){
            if (pid->at(i)==11 || pid->at(i)==13){
                lepton_idx=i;
            }
        }
    }

    *minDeltaR = 999;
    float deltaR;
    fastjet::PseudoJet lepton = fj[lepton_idx];
    for (int i=0; i<fj.size(); i++){
        if (i==lepton_idx) continue;
        deltaR = lepton.delta_R(fj[i]);
        if (deltaR < *minDeltaR) *minDeltaR=deltaR;
    }

    //std::cout << "Lepton is Isolated: " << isIsolated << std::endl;

    return lepton_idx;
}
