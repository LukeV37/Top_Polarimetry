float calc_costheta(TLorentzVector p_t, TLorentzVector p_tbar, TLorentzVector p_d){
    float costheta;
    TVector3 to_ttbar_rest;
    TVector3 to_t_rest;
    TVector3 k_vect;
    TVector3 d_vect;

    // Construct Lorentz boost to t-tbar CM frame
    to_ttbar_rest = -(p_t + p_tbar).BoostVector();

    // Boost vectors to t-tbar CM frame
    p_t.Boost(to_ttbar_rest);
    p_tbar.Boost(to_ttbar_rest);
    p_d.Boost(to_ttbar_rest);

    // Top quark unit vector in t-tbar CM frame
    k_vect = p_t.Vect().Unit();

    // Construct Lorentz boos to top quark rest frame
    to_t_rest = -p_t.BoostVector();

    // Boost down quark to top quark rest frame
    p_d.Boost(to_t_rest);

    // Down quark unit vector in top rest frame
    d_vect = p_d.Vect().Unit();

    // Calc dot product for cos theta
    costheta = k_vect.Dot(d_vect);

    return costheta;
}
