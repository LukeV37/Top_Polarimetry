float calc_costheta(TLorentzVector p_t, TLorentzVector p_tbar, TLorentzVector p_d, float *top_px_boosted, float *top_py_boosted, float *top_pz_boosted, float *top_e_boosted, float *down_px_boosted, float *down_py_boosted, float *down_pz_boosted, float *down_e_boosted){
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

    // Store top quark kinematics in t-tbar CM frame
    *top_px_boosted = p_t.Px();
    *top_py_boosted = p_t.Py();
    *top_pz_boosted = p_t.Pz();
    *top_e_boosted = p_t.E();

    // Top quark unit vector in t-tbar CM frame
    k_vect = p_t.Vect().Unit();

    // Construct Lorentz boos to top quark rest frame
    to_t_rest = -p_t.BoostVector();

    // Boost down quark to top quark rest frame
    p_d.Boost(to_t_rest);

    // Store down quark kinematics in top rest frame
    *down_px_boosted = p_d.Px();
    *down_py_boosted = p_d.Py();
    *down_pz_boosted = p_d.Pz();
    *down_e_boosted = p_d.E();

    // Down quark unit vector in top rest frame
    d_vect = p_d.Vect().Unit();

    // Calc dot product for cos theta
    costheta = k_vect.Dot(d_vect);

    return costheta;
}
