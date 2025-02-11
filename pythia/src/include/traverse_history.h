int find_top_from_event(const Pythia8::Event& event){
    // Iterate through particles and analyze partons
    for(int i=0;i<event.size();i++){
        auto &p = event[i];

        if(p.id()!=6) continue;

        // Get top daughters and look for W+ and b
        int d1 = p.daughter1();
        int d2 = p.daughter2();

        if(event[d1].id()==6 || event[d2].id()==6) continue; // Skip ISR; go until top decays

        // Loop over top daughters; look for b quark
        for (int j=d1; j<=d2; j++){
            if(event[j].id()==5) {
                return i; // If top decayed to b, return idx
            }
        }
    }
    return -1; // Nothing good
}

int find_down_from_W(const Pythia8::Event& event, int W_idx){
    auto &Wboson = event[W_idx];

    int d1 = Wboson.daughter1();
    int d2 = Wboson.daughter2();

    if (event[d1].id()==24) return find_down_from_W(event, d1);

    for (int i=d1; d1<=d2; i++){
        if (event[i].idAbs()==1 || event[i].idAbs()==3){
            return i;
        }
    }
    return -1; // Nothing good
}

int find_down_from_top(const Pythia8::Event& event, int top_idx){
    auto &top = event[top_idx];

    int d1 = top.daughter1();
    int d2 = top.daughter2();

    for (int i=d1; d1<=d2; i++){
        // Find down type quark through W boson
        if (event[i].id()==24){
            return find_down_from_W(event, i);
        }
        // Find down quark directly from top
        if (event[i].idAbs()==1 || event[i].idAbs()==3){
            return i;
        }
    }
    return -1; // Nothing good
}

void traverse_history(const Pythia8::Event& event, std::vector<int> &fromMother, int current_idx){
    // Flag current particle as from down
    fromMother[current_idx] = 1;

    // Traverse Daughters
    int d1 = event[current_idx].daughter1();
    int d2 = event[current_idx].daughter2();

    // Normal case where daughters are stored sequentially
    if (d1<=d2 && d1>0){
        for (int i=d1; d1<=d2; i++){
            if (i>d2) return; // Needed for recursion termination; otherwise odd behavior
            if (fromMother[i]==0) {
                //std::cout << "RECURSIVE CONDITION:\t" << current_idx << "\t" << d1 << "\t" << d2 << std::endl;
                traverse_history(event, fromMother, i);
            }
        }
    }

    // Special case where two daughters are stored not sequentially
    if (d2<d1 && d2>0){
        for (int i=0; i<2; i++){
            if (fromMother[i]==0) {
                //std::cout << "RECURSIVE CONDITION:\t" << current_idx << "\t" << d1 << "\t" << d2 << std::endl;
                if (i==0) traverse_history(event, fromMother, d1);
                if (i==1) traverse_history(event, fromMother, d2);
                if (i>=2) return;
            }
        }
    }

    // Special case where d1>0 && d2=0
    if (d1>0 && d2==0){
        if (fromMother[d1]==0) {
            //std::cout << "RECURSIVE CONDITION:\t" << current_idx << "\t" << d1 << "\t" << d2 << std::endl;
            return traverse_history(event, fromMother, d1);
        }
    }

    // There are no more daughters
    if ( (d1==0) && (d2==0) ) {
        //std::cout << "BASE CONDITION:\t" << current_idx << "\t" << d1 << "\t" << d2 << std::endl;
        return;
    }
}

std::vector<int> find_daughters(const Pythia8::Event& event, int idx){
    //Initialize fromMother vector
    std::vector<int> fromMother(event.size(), 0);

    // Traverse Down History
    traverse_history(event, fromMother, idx);

    return fromMother;
}
