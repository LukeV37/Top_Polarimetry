int get_MET(std::vector<fastjet::PseudoJet> fj, std::vector<int> *fromNu, std::vector<int> *PID)
{
    int nu_idx=-1;

    for (int i=0; i<fromNu->size(); i++){
        if (fromNu->at(i)==1){
            nu_idx=i;
        }
    }

    return nu_idx;
}
