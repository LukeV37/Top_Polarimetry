void find_ip(double pT, double eta, double phi, double xProd, double yProd, double zProd, double& d0, double& z0)
{
  // calculate IP

  double r = sqrt(pow(xProd,2) + pow(yProd,2));
  double dphi = phi - atan2(yProd,xProd);
  if (dphi>M_PI) dphi -= 2*M_PI;
  else if (dphi<=-M_PI) dphi += 2*M_PI;
  d0 = r*sin(dphi);
  z0 = zProd - r*sinh(eta);

  // smear according to resolution
  // tentative resolution parameterization: ATL-COM-PHYS-2021-377, Figs.12-13

  double sigma_d0 = sqrt(pow(80/pT,2) + pow(4,2))*1e-3; // um->mm
  double sigma_z0 = sqrt(pow(80/pT,2) + pow(10,2))*1e-3; // um->mm

  static TRandom3 rnd;
  d0 += rnd.Gaus(0,sigma_d0);
  z0 += rnd.Gaus(0,sigma_z0);
}
