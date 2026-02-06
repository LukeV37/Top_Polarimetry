import ROOT
from ROOT import TLorentzVector, TVector3
import numpy as np
import uproot
import awkward as ak
import sys

class Event:
    def __init__(self):
        # Inputs
        self.b = self.particle()
        self.u = self.particle()
        self.d = self.particle()
        self.bbar = self.particle()
        self.lep = self.particle()
        self.nu = self.particle()
        
        # Labels
        self.top_label = self.particle()
        self.down_label = self.particle()
        self.b_label = self.particle()
        self.costheta = []
        
    def __len__(self):
        length = len(self.b.px)
        if (len(self.u.px)!=length 
            or len(self.d.px)!=length 
            or len(self.bbar.px)!=length 
            or len(self.lep.px)!=length 
            or len(self.nu.px)!=length):
            raise Exception("Event does not contain uniquely identified quarks!"
                            +"Please refine selections.")
        return length
        
    class particle:
        def __init__(self):
            self.px = []
            self.py = []
            self.pz = []
            self.E = []
            self.pT = []
            self.eta = []
            self.phi = []
            self.deltaR = []
            self.deltaEta = []
            self.deltaPhi = []
            
        def add_feats(self, px, py, pz, E):
            self.px = np.concat((self.px, ak.to_numpy(px)))
            self.py = np.concat((self.py, ak.to_numpy(py)))
            self.pz = np.concat((self.pz, ak.to_numpy(pz)))
            self.E = np.concat((self.E, ak.to_numpy(E)))
            
    def calc_labels(self):
        # Initialize Labels
        self.top_label.px = np.ones_like(self.b.px)*-999
        self.top_label.py = np.ones_like(self.b.px)*-999
        self.top_label.pz = np.ones_like(self.b.px)*-999
        self.top_label.E = np.ones_like(self.b.px)*-999
        self.top_label.pT = np.ones_like(self.b.px)*-999
        self.top_label.eta = np.ones_like(self.b.px)*-999
        self.top_label.phi = np.ones_like(self.b.px)*-999
        
        self.down_label.px = np.ones_like(self.b.px)*-999
        self.down_label.py = np.ones_like(self.b.px)*-999
        self.down_label.pz = np.ones_like(self.b.px)*-999
        self.down_label.E = np.ones_like(self.b.px)*-999
        self.down_label.pT = np.ones_like(self.b.px)*-999
        self.down_label.eta = np.ones_like(self.b.px)*-999
        self.down_label.phi = np.ones_like(self.b.px)*-999
        self.down_label.deltaR = np.ones_like(self.b.px)*-999
        self.down_label.deltaEta = np.ones_like(self.b.px)*-999
        self.down_label.deltaPhi = np.ones_like(self.b.px)*-999

        self.b_label.px = np.ones_like(self.b.px)*-999
        self.b_label.py = np.ones_like(self.b.px)*-999
        self.b_label.pz = np.ones_like(self.b.px)*-999
        self.b_label.E = np.ones_like(self.b.px)*-999
        self.b_label.pT = np.ones_like(self.b.px)*-999
        self.b_label.eta = np.ones_like(self.b.px)*-999
        self.b_label.phi = np.ones_like(self.b.px)*-999
        self.b_label.deltaR = np.ones_like(self.b.px)*-999
        self.b_label.deltaEta = np.ones_like(self.b.px)*-999
        self.b_label.deltaPhi = np.ones_like(self.b.px)*-999
        
        self.costheta = np.ones_like(self.b.px)*-999
        
        # Loop over events and calculate labels
        for event in range(len(self)):
            #Define particle's 4-vectors (px,py,pz,E)
            p_b = TLorentzVector(self.b.px[event],self.b.py[event],self.b.pz[event],self.b.E[event])
            p_d = TLorentzVector(self.d.px[event],self.d.py[event],self.d.pz[event],self.d.E[event])
            p_u = TLorentzVector(self.u.px[event],self.u.py[event],self.u.pz[event],self.u.E[event])

            p_bbar = TLorentzVector(self.bbar.px[event],self.bbar.py[event],self.bbar.pz[event],self.bbar.E[event])
            p_lm = TLorentzVector(self.lep.px[event],self.lep.py[event],self.lep.pz[event],self.lep.E[event])
            p_vl = TLorentzVector(self.nu.px[event],self.nu.py[event],self.nu.pz[event],self.nu.E[event])

            #Construct tops in lab frame by adding the corresponding decay products
            p_t = p_b + p_d + p_u 
            p_tbar = p_bbar + p_lm + p_vl

            #Construct Lorentz boost to t-tbar CM frame 
            to_ttbar_rest = -(p_t + p_tbar).BoostVector()

            #Boost vectors to t-tbar CM frame
            p_t.Boost(to_ttbar_rest)
            p_tbar.Boost(to_ttbar_rest)
            p_d.Boost(to_ttbar_rest)
            p_b.Boost(to_ttbar_rest)

            # Store top quark kinematics in ttbar CM frame
            self.top_label.px[event] = p_t.Px()
            self.top_label.py[event] = p_t.Py()
            self.top_label.pz[event] = p_t.Pz()
            self.top_label.E[event]  = p_t.E()
            self.top_label.pT[event] = p_t.Pt()
            self.top_label.eta[event] = p_t.Eta()
            self.top_label.phi[event] = p_t.Phi()

            #Top quark direction in t-tbar CM frame
            k_vect = p_t.Vect().Unit()

            #Define boost from t-tbar CM frame to top rest frame
            to_t_rest = -p_t.BoostVector()

            #Boost down quark to top quark rest frame
            p_d.Boost(to_t_rest)
            p_b.Boost(to_t_rest)

            # Store bottom quark kinematics in ttbar rest frame
            norm = np.sqrt(p_b.Px()**2 + p_b.Py()**2 + p_b.Pz()**2)
            self.b_label.px[event] = p_b.Px()/norm
            self.b_label.py[event] = p_b.Py()/norm
            self.b_label.pz[event] = p_b.Pz()/norm
            self.b_label.E[event]  = p_b.E()
            self.b_label.pT[event] = p_b.Pt()
            self.b_label.eta[event] = p_b.Eta()
            self.b_label.phi[event] = p_b.Phi()
            self.b_label.deltaR[event] = p_b.DeltaR(p_t)
            self.b_label.deltaEta[event] = p_b.Eta()-p_t.Eta()
            self.b_label.deltaPhi[event] = p_b.DeltaPhi(p_t)

            # Store down quark kinematics in t rest frame
            norm = np.sqrt(p_d.Px()**2 + p_d.Py()**2 + p_d.Pz()**2)
            self.down_label.px[event] = p_d.Px()/norm
            self.down_label.py[event] = p_d.Py()/norm
            self.down_label.pz[event] = p_d.Pz()/norm
            self.down_label.E[event]  = p_d.E()            
            self.down_label.pT[event] = p_d.Pt()
            self.down_label.eta[event] = p_d.Eta()
            self.down_label.phi[event] = p_d.Phi()
            self.down_label.deltaR[event] = p_d.DeltaR(p_t)
            self.down_label.deltaEta[event] = p_d.Eta()-p_t.Eta()
            self.down_label.deltaPhi[event] = p_d.DeltaPhi(p_t)

            #down quark direction in top rest frame
            d_vect = p_d.Vect().Unit()

            #\cos\theta_d = k_vect\dot d_vect
            self.costheta[event] = k_vect.Dot(d_vect)
            
    def write_ntuple(self, tag, num):
        with uproot.recreate("pp_tt_semi_full_"+tag+"/labels_"+tag+"_"+num+".root") as f:
            tree = f.mktree("labels", {"top_px": "float32",
                                       "top_py": "float32",
                                       "top_pz": "float32",
                                       "top_E": "float32",
                                       "top_pT": "float32",
                                       "top_eta": "float32",
                                       "top_phi": "float32",
                                       "down_px": "float32",
                                       "down_py": "float32",
                                       "down_pz": "float32",
                                       "down_E": "float32",
                                       "down_pT": "float32",
                                       "down_eta": "float32",
                                       "down_phi": "float32",
                                       "down_deltaR": "float32",
                                       "down_deltaEta": "float32",
                                       "down_deltaPhi": "float32",
                                       "bottom_px": "float32",
                                       "bottom_py": "float32",
                                       "bottom_pz": "float32",
                                       "bottom_E": "float32",
                                       "bottom_pT": "float32",
                                       "bottom_eta": "float32",
                                       "bottom_phi": "float32",
                                       "bottom_deltaR": "float32",
                                       "bottom_deltaEta": "float32",
                                       "bottom_deltaPhi": "float32",
                                       "costheta": "float32",
                                       }
                           )
            tree.extend( {"top_px": self.top_label.px,
                          "top_py": self.top_label.py,
                          "top_pz": self.top_label.pz,
                          "top_E" : self.top_label.E,
                          "top_pT" : self.top_label.pT,
                          "top_eta" : self.top_label.eta,
                          "top_phi" : self.top_label.phi,
                          "down_px": self.down_label.px,
                          "down_py": self.down_label.py,
                          "down_pz": self.down_label.pz,
                          "down_E" : self.down_label.E,
                          "down_pT" : self.down_label.pT,
                          "down_eta" : self.down_label.eta,
                          "down_phi" : self.down_label.phi,
                          "down_deltaR" : self.down_label.deltaR,
                          "down_deltaEta" : self.down_label.deltaEta,
                          "down_deltaPhi" : self.down_label.deltaPhi,
                          "bottom_px": self.b_label.px,
                          "bottom_py": self.b_label.py,
                          "bottom_pz": self.b_label.pz,
                          "bottom_E" : self.b_label.E,
                          "bottom_pT" : self.b_label.pT,
                          "bottom_eta" : self.b_label.eta,
                          "bottom_phi" : self.b_label.phi,
                          "bottom_deltaR" : self.b_label.deltaR,
                          "bottom_deltaEta" : self.b_label.deltaEta,
                          "bottom_deltaPhi" : self.b_label.deltaPhi,
                          "costheta": self.costheta
                         }
                       )

# Get dataset tag
dataset_tag=str(sys.argv[1])
run_num=str(sys.argv[2])

# Read input ntuple
#print("\tReading lhe file...")
with uproot.open('pp_tt_semi_full_'+dataset_tag+'/hard_process_'+dataset_tag+'_'+run_num+'.root:events') as f:
    #print(f.keys())
    px = f['px'].array()
    py = f['py'].array()
    pz = f['pz'].array()
    E  = f['energy'].array()
    pid= f['pid'].array()
    m1 = f['mother1'].array()
    m2 = f['mother2'].array()
    status = f['status'].array()

# Initialize Event object
event = Event()

# Generate down masks
d_mask = (abs(pid)==1)
s_mask = (abs(pid)==3)
status = status>0
# Particle can be down OR strange AND must not be parton of incoming proton
mask = (d_mask | s_mask) & status
event.d.add_feats(np.ravel(px[mask]),np.ravel(py[mask]),np.ravel(pz[mask]),np.ravel(E[mask]))

# Generate up masks
u_mask = (abs(pid)==2)
c_mask = (abs(pid)==4)
status = status>0
# Particle can be up OR charm AND must not be parton of incoming proton
mask = (u_mask | c_mask) & status
event.u.add_feats(np.ravel(px[mask]),np.ravel(py[mask]),np.ravel(pz[mask]),np.ravel(E[mask]))

# Generate b masks
b_mask = pid==5
status = status>0
# Particle is bottom and must not be parton of incoming proton
mask = b_mask & status
event.b.add_feats(np.ravel(px[mask]),np.ravel(py[mask]),np.ravel(pz[mask]),np.ravel(E[mask]))

# Generate anti-b masks
b_mask = pid==-5
status = status>0
# Particle is anti-bottom and must not be parton of incoming proton
mask = b_mask & status
event.bbar.add_feats(np.ravel(px[mask]),np.ravel(py[mask]),np.ravel(pz[mask]),np.ravel(E[mask]))

# Generate lepton masks
el_mask = (abs(pid)==11)
mu_mask = (abs(pid)==13)
# Particle can be electron OR muon
mask = (el_mask | mu_mask)
event.lep.add_feats(np.ravel(px[mask]),np.ravel(py[mask]),np.ravel(pz[mask]),np.ravel(E[mask]))

# Generate neutrino masks
nu_el_mask = (abs(pid)==12)
nu_mu_mask = (abs(pid)==14)
# Particle can be nu_el OR nu_mu
mask = (nu_el_mask | nu_mu_mask)
event.nu.add_feats(np.ravel(px[mask]),np.ravel(py[mask]),np.ravel(pz[mask]),np.ravel(E[mask]))

# Calculate and write labels
#print("\tCalculating Labels...")
event.calc_labels()
#print("\tWriting ntuple...")
event.write_ntuple(dataset_tag,run_num)
#print("\tDone!")
