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
        
        # Lables
        self.top_label = self.particle()
        self.down_label = self.particle()
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
            
        def add_feats(self, px, py, pz, E):
            self.px = np.concat((self.px, ak.to_numpy(px)))
            self.py = np.concat((self.py, ak.to_numpy(py)))
            self.pz = np.concat((self.pz, ak.to_numpy(pz)))
            self.E = np.concat((self.E, ak.to_numpy(E)))
            
    def calc_labels(self):
        # Initialize Labels
        self.top_label.px = np.ones_like(self.b.px)*-999
        self.top_label.py = np.ones_like(self.b.py)*-999
        self.top_label.pz = np.ones_like(self.b.pz)*-999
        self.top_label.E = np.ones_like(self.b.E)*-999
        
        self.down_label.px = np.ones_like(self.b.px)*-999
        self.down_label.py = np.ones_like(self.b.py)*-999
        self.down_label.pz = np.ones_like(self.b.pz)*-999
        self.down_label.E = np.ones_like(self.b.E)*-999
        
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
            
            # Store top quark kinematics in ttbar CM frame
            self.top_label.px[event] = p_t.Px()
            self.top_label.py[event] = p_t.Py()
            self.top_label.pz[event] = p_t.Pz()
            self.top_label.E[event]  = p_t.E()

            #Top quark direction in t-tbar CM frame
            k_vect = p_t.Vect().Unit()

            #Define boost from t-tbar CM frame to top rest frame
            to_t_rest = -p_t.BoostVector()

            #Boost down quark to top quark rest frame
            p_d.Boost(to_t_rest)
            
            # Store down quark kinematics in t rest frame
            self.down_label.px[event] = p_d.Px()
            self.down_label.py[event] = p_d.Py()
            self.down_label.pz[event] = p_d.Pz()
            self.down_label.E[event]  = p_d.E()            

            #down quark direction in top rest frame
            d_vect = p_d.Vect().Unit()

            #\cos\theta_d = k_vect\dot d_vect
            self.costheta[event] = k_vect.Dot(d_vect)
            
    def write_ntuple(self, tag):
        with uproot.recreate("pp_tt_semi_full_"+tag+"/labels_"+tag+".root") as f:
            f['labels'] = {"top_px": self.top_label.px,
                          "top_py": self.top_label.py,
                          "top_pz": self.top_label.pz,
                          "top_E" : self.top_label.E,
                          "down_px": self.down_label.px,
                          "down_py": self.down_label.py,
                          "down_pz": self.down_label.pz,
                          "down_E" : self.down_label.E,
                          "costheta": self.costheta
                         }
            #f['labels'].show()

# Get dataset tag
dataset_tag=str(sys.argv[1])

# Read input ntuple
print("Reading lhe file...")
with uproot.open('pp_tt_semi_full_'+dataset_tag+'/hard_process_'+dataset_tag+'.root:events') as f:
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
print("Calculating Labels...")
event.calc_labels()
print("Writing ntuple...")
event.write_ntuple(dataset_tag)
print("Done!")
