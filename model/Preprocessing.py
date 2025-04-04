import uproot
import awkward as ak
import numpy as np
import vector
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score
import pickle
import sys

try:
    _=sys.argv[5]
except:
    print("Must enter 5 agruments")
    print("\t1: dataset tag (from MadGraph)")
    print("\t2: number of runs (from MadGraph)")
    print("\t3: plotting output directory e.g. plots")
    print("\t4: dataset output directory e.g data")
    print("\t5: analysis type e.g. bottom or down")
    exit(1)

dataset_tag = str(sys.argv[1])
run_num = str(sys.argv[2])
out_dir_plots = str(sys.argv[3])
out_dir_data = str(sys.argv[4])
analysis_type = str(sys.argv[5])

print("Reading Sample...")

# Open Pythia file
with uproot.open("../pythia/WS_"+dataset_tag+"/dataset_"+dataset_tag+"_"+run_num+".root:fastjet") as f:
    #print(f.keys())
    jet_pt = f['jet_pt'].array()
    jet_eta = f['jet_eta'].array()
    jet_phi = f['jet_phi'].array()
    jet_m = f['jet_m'].array()
    
    jet_trk_pt = f['jet_trk_pt'].array()
    jet_trk_eta = f['jet_trk_eta'].array()
    jet_trk_phi = f['jet_trk_phi'].array()
    jet_trk_q = f['jet_trk_q'].array()
    jet_trk_d0 = f['jet_trk_d0'].array()
    jet_trk_z0 = f['jet_trk_z0'].array()
    #jet_trk_pid = f['jet_trk_pid'].array()
    jet_trk_origin = f['jet_trk_origin'].array()
    jet_trk_fromDown = f['jet_trk_fromDown'].array()
    jet_trk_fromUp = f['jet_trk_fromUp'].array()
    jet_trk_fromBottom = f['jet_trk_fromBottom'].array()
    
    trk_pt = f['trk_pt'].array()
    trk_eta = f['trk_eta'].array()
    trk_phi = f['trk_phi'].array()
    trk_q = f['trk_q'].array()
    trk_d0 = f['trk_d0'].array()
    trk_z0 = f['trk_z0'].array()
    #trk_pid = f['trk_pid'].array()
    #trk_origin = f['trk_origin'].array()
    #trk_fromDown = f['trk_fromDown'].array()
    #trk_fromUp = f['trk_fromUp'].array()
    #trk_fromBottom = f['trk_fromBottom'].array()

with uproot.open("../madgraph/pp_tt_semi_full_"+dataset_tag+"/labels_"+dataset_tag+"_"+run_num+".root:labels") as f:
    #print(f.keys())
    top_px = f['top_px'].array()
    top_py = f['top_py'].array()
    top_pz = f['top_pz'].array()
    top_E = f['top_E'].array()
    top_pT = f['top_pT'].array()
    top_eta = f['top_eta'].array()
    top_phi = f['top_phi'].array()
    down_px = f['down_px'].array()
    down_py = f['down_py'].array()
    down_pz = f['down_pz'].array()
    down_pT = f['down_pT'].array()
    down_eta = f['down_eta'].array()
    down_phi = f['down_phi'].array()
    down_deltaR = f['down_deltaR'].array()
    down_deltaEta = f['down_deltaEta'].array()
    down_deltaPhi = f['down_deltaPhi'].array()
    bottom_px = f['bottom_px'].array()
    bottom_py = f['bottom_py'].array()
    bottom_pz = f['bottom_pz'].array()
    bottom_pT = f['bottom_pT'].array()
    bottom_eta = f['bottom_eta'].array()
    bottom_phi = f['bottom_phi'].array()
    bottom_deltaR = f['bottom_deltaR'].array()
    bottom_deltaEta = f['bottom_deltaEta'].array()
    bottom_deltaPhi = f['bottom_deltaPhi'].array()
    costheta = f['costheta'].array()

# Initialize output features
selected_events = []

# Jet Feats
selected_jet_pt = []
selected_jet_eta = []
selected_jet_phi = []
selected_jet_m = []

# Trk Feats
selected_trk_pt = []
selected_trk_eta = []
selected_trk_phi = []
selected_trk_q = []
selected_trk_d0 = []
selected_trk_z0 = []
#selected_trk_pid = []
#selected_trk_origin = []
selected_trk_fromDown = []
selected_trk_fromUp = []
selected_trk_fromBottom = []

deltaR_cut = 1.0

# Initialize lists for plotting
pt_partons = []
pt_fat_jets = []
deltaR = []
deltaEta = []
deltaPhi = []

unweighted_origins = []
weighted_origins = []

missing_jet=0
cutflow_deltaR=0

# Loop over all events
num_events=len(jet_pt)
for i in range(num_events):

    mod=100
    if i%mod==0:
        print("\tProcessing: ", i+mod, " / ", len(jet_pt), end="\r")

    # Ensure at least one fat jet
    if len(jet_pt[i])<1:
        missing_jet+=1
        selected_events.append(False)
        continue

    # Save parton 3-Vector
    parton = vector.MomentumObject3D(px=top_px[i], py=top_py[i], pz=top_pz[i])

    # Find closest fat jet in deltaR
    delR = []
    for j in range(len(jet_pt[i])):
        delR.append(parton.deltaR(vector.MomentumObject3D(pt=jet_pt[i][j], eta=jet_eta[i][j], phi=jet_phi[i][j])))
    argmin = np.argmin(delR)

    candidate = vector.MomentumObject3D(pt=jet_pt[i][argmin], eta=jet_eta[i][argmin], phi=jet_phi[i][argmin])

    if abs(parton.deltaR(candidate))>deltaR_cut:
        cutflow_deltaR+=1
        selected_events.append(False)
        continue

    # Closest jet in deltaR to parton is matched jet
    matched_jet = candidate
    selected_events.append(True)

    # Fill lists for plotting
    pt_partons.append(parton.pt)
    pt_fat_jets.append(matched_jet.pt)
    deltaR.append(parton.deltaR(matched_jet))
    deltaEta.append(parton.deltaeta(matched_jet))
    deltaPhi.append(parton.deltaphi(matched_jet))
    
    # Calculate Truth Origins
    weights = jet_trk_pt[i][argmin]
    origins = jet_trk_origin[i][argmin]
    fromTop = origins==6 
    fromW = origins==24
    fromHadronic=fromTop|fromW
    unweighted = ak.mean(fromHadronic,axis=0)
    weighted = ak.mean(fromHadronic,axis=0,weight=weights)
    unweighted_origins.append(unweighted)
    weighted_origins.append(weighted)
    
    # Fill output vars
    # Jet Feats
    selected_jet_pt.append(jet_pt[i][argmin])
    selected_jet_eta.append(jet_eta[i][argmin])
    selected_jet_phi.append(jet_phi[i][argmin])
    selected_jet_m.append(jet_m[i][argmin])

    # Trk Feats
    selected_trk_pt.append(jet_trk_pt[i][argmin])
    selected_trk_eta.append(jet_trk_eta[i][argmin])
    selected_trk_phi.append(jet_trk_phi[i][argmin])
    selected_trk_q.append(jet_trk_q[i][argmin])
    selected_trk_d0.append(jet_trk_d0[i][argmin])
    selected_trk_z0.append(jet_trk_z0[i][argmin])
    #selected_trk_pid.append(jet_trk_pid[i][argmin])
    #selected_trk_origin.append(jet_trk_origin[i][argmin])
    selected_trk_fromDown.append(jet_trk_fromDown[i][argmin])
    selected_trk_fromUp.append(jet_trk_fromUp[i][argmin])
    selected_trk_fromBottom.append(jet_trk_fromBottom[i][argmin])

print()    
print("\tEvents without reco jet: ", missing_jet, "/", num_events)
print("\tDeltaR Cutflow: ", cutflow_deltaR, "/", num_events-missing_jet)
print()

plt.figure()
plt.title("Fraction of Tracks Originating From Top or W+")
plt.hist(unweighted_origins,bins=50,range=(0,1),color='r',histtype='step',label="Unweighted")
plt.hist(weighted_origins,bins=50,range=(0,1),color='b',histtype='step',label="Weighted by pT")
plt.yscale('log')
plt.ylabel('Num Jets')
plt.xlabel('Fraction of Tracks From Top',loc='right')
plt.legend()
plt.savefig(out_dir_plots+"/run_"+run_num+"/Fraction_from_Top.png")

fig1, ax1 = plt.subplots()
ax1.set_title("Matched Jet vs Parton pT")
ax1.hist2d(pt_partons,pt_fat_jets, bins=100,norm=mcolors.LogNorm(),range=((200,800),(200,800)))
ax1.set_xlabel("Parton pT")
ax1.set_ylabel("Fat Jet pT")
ax1.text(600,300,"$R^2$ value: "+str(round(r2_score(pt_partons,pt_fat_jets),3)),backgroundcolor='r',color='k')
plt.savefig(out_dir_plots+"/run_"+run_num+"/Matching_pT.png")

fig2, ax2 = plt.subplots()
ax2.set_title("DeltaR between Matched Jet and Parton")
ax2.hist(deltaR,histtype='step',bins=100,range=(0,4),color='k')
ax2.set_yscale("log")
#ax2.legend()
plt.savefig(out_dir_plots+"/run_"+run_num+"/Matching_DeltaR.png")

fig3, ax3 = plt.subplots()
ax3.set_title("DeltaEta between Matched Jet and Parton")
ax3.hist(deltaEta,histtype='step',bins=100,range=(-3.5,3.5),color='k')
ax3.set_yscale("log")
#ax3.legend()
plt.savefig(out_dir_plots+"/run_"+run_num+"/Matching_DeltaEta.png")

fig4, ax4 = plt.subplots()
ax4.set_title("DeltaPhi between Matched Jet and Parton")
ax4.hist(deltaPhi,histtype='step',bins=100,range=(-3.5,3.5),color='k')
ax4.set_yscale("log")
#ax4.legend()
plt.savefig(out_dir_plots+"/run_"+run_num+"/Matching_DeltaPhi.png")
    

print("Converting to Awkward Arrays...")

# Jet Feats
jet_pt = ak.Array(selected_jet_pt)
jet_eta = ak.Array(selected_jet_eta)
jet_phi = ak.Array(selected_jet_phi)
jet_m = ak.Array(selected_jet_m)

# Trk Feats
jet_trk_pt = ak.Array(selected_trk_pt)
jet_trk_eta = ak.Array(selected_trk_eta)
jet_trk_phi = ak.Array(selected_trk_phi)
jet_trk_q = ak.Array(selected_trk_q)
jet_trk_d0 = ak.Array(selected_trk_d0)
jet_trk_z0 = ak.Array(selected_trk_z0)
#jet_trk_pid = ak.Array(selected_trk_pid)
#jet_trk_origin = ak.Array(selected_trk_origin)
jet_trk_fromDown = ak.Array(selected_trk_fromDown)
jet_trk_fromUp = ak.Array(selected_trk_fromUp)
jet_trk_fromBottom = ak.Array(selected_trk_fromBottom)

# Global Trk Feats
trk_pt = trk_pt[selected_events]
trk_eta = trk_eta[selected_events]
trk_phi = trk_phi[selected_events]
trk_q = trk_q[selected_events]
trk_d0 = trk_d0[selected_events]
trk_z0 = trk_z0[selected_events]
#trk_pid = trk_pid[selected_events]
#trk_origin = trk_origin[selected_events]
#trk_fromDown = trk_fromDown[selected_events]

# Label
top_px = top_px[selected_events]
top_py = top_py[selected_events]
top_pz = top_pz[selected_events]
top_E = top_E[selected_events]
down_px = down_px[selected_events]
down_py = down_py[selected_events]
down_pz = down_pz[selected_events]
down_pT = down_pT[selected_events]
down_eta = down_eta[selected_events]
down_phi = down_phi[selected_events]
down_deltaR = down_deltaR[selected_events]
down_deltaEta = down_deltaEta[selected_events]
down_deltaPhi = down_deltaPhi[selected_events]
bottom_px = bottom_px[selected_events]
bottom_py = bottom_py[selected_events]
bottom_pz = bottom_pz[selected_events]
bottom_pT = bottom_pT[selected_events]
bottom_eta = bottom_eta[selected_events]
bottom_phi = bottom_phi[selected_events]
bottom_deltaR = bottom_deltaR[selected_events]
bottom_deltaEta = bottom_deltaEta[selected_events]
bottom_deltaPhi = bottom_deltaPhi[selected_events]
costheta = costheta[selected_events]

norm = False
if norm:
    jet_pt = (jet_pt-ak.mean(jet_pt))/ak.std(jet_pt)
    jet_eta = (jet_eta-ak.mean(jet_eta))/ak.std(jet_eta)
    jet_phi = (jet_phi-ak.mean(jet_phi))/ak.std(jet_phi)
    jet_m = (jet_m-ak.mean(jet_m))/ak.std(jet_m)
    
    jet_trk_pt = (jet_trk_pt-ak.mean(jet_trk_pt))/ak.std(jet_trk_pt)
    jet_trk_eta = (jet_trk_eta-ak.mean(jet_trk_eta))/ak.std(jet_trk_eta)
    jet_trk_phi = (jet_trk_phi-ak.mean(jet_trk_phi))/ak.std(jet_trk_phi)
    jet_trk_q = (jet_trk_q-ak.mean(jet_trk_q))/ak.std(jet_trk_q)
    jet_trk_d0 = (jet_trk_d0-ak.mean(jet_trk_d0))/ak.std(jet_trk_d0)
    jet_trk_z0 = (jet_trk_z0-ak.mean(jet_trk_z0))/ak.std(jet_trk_z0)
    
    trk_pt = (trk_pt-ak.mean(trk_pt))/ak.std(trk_pt)
    trk_eta = (trk_eta-ak.mean(trk_eta))/ak.std(trk_eta)
    trk_phi = (trk_phi-ak.mean(trk_phi))/ak.std(trk_phi)
    trk_q = (trk_q-ak.mean(trk_q))/ak.std(trk_q)
    trk_d0 = (trk_d0-ak.mean(trk_d0))/ak.std(trk_d0)
    trk_z0 = (trk_z0-ak.mean(trk_z0))/ak.std(trk_z0)

jet_feat_list = [jet_pt,jet_eta,jet_phi,jet_m]
jet_feat_list = [x[:,np.newaxis] for x in jet_feat_list]
jet_feats = ak.concatenate(jet_feat_list, axis=1)

#jet_trk_feat_list = [jet_trk_pt,jet_trk_eta,jet_trk_phi,jet_trk_q,jet_trk_d0,jet_trk_z0,jet_trk_pid,jet_trk_origin,jet_trk_fromDown]
#jet_trk_feat_list = [jet_trk_pt,jet_trk_eta,jet_trk_phi,jet_trk_q,jet_trk_d0,jet_trk_z0,jet_trk_pid,jet_trk_fromDown]
jet_trk_feat_list = [jet_trk_pt,jet_trk_eta,jet_trk_phi,jet_trk_q,jet_trk_d0,jet_trk_z0,jet_trk_fromDown,jet_trk_fromUp,jet_trk_fromBottom]
jet_trk_feat_list = [x[:,:,np.newaxis] for x in jet_trk_feat_list]
jet_trk_feats = ak.concatenate(jet_trk_feat_list, axis=2)

#trk_feat_list = [trk_pt,trk_eta,trk_phi,trk_q,trk_d0,trk_z0,trk_pid,trk_origin,trk_fromDown]
#trk_feat_list = [trk_pt,trk_eta,trk_phi,trk_q,trk_d0,trk_z0,trk_pid,trk_fromDown]
trk_feat_list = [trk_pt,trk_eta,trk_phi,trk_q,trk_d0,trk_z0]
trk_feat_list = [x[:,:,np.newaxis] for x in trk_feat_list]
trk_feats = ak.concatenate(trk_feat_list, axis=2)

#label_list = [top_px,top_py,top_pz,top_E,down_px,down_py,down_pz,down_pT,down_eta,down_phi,down_deltaR,down_deltaEta,down_deltaPhi,bottom_px,bottom_py,bottom_pz,bottom_pT,bottom_eta,bottom_phi,bottom_deltaR,bottom_deltaEta,bottom_deltaPhi,costheta]

assert analysis_type=="bottom" or analysis_type=="down"

if analysis_type=="bottom":
    label_list = [bottom_px,bottom_py,bottom_pz]
if analysis_type=="down":
    label_list = [down_px,down_py,down_pz]

label_list = [x[:,np.newaxis] for x in label_list]
labels = ak.concatenate(label_list, axis=1)

data_dict = {"jet_feats": jet_feats,
             "jet_trk_feats": jet_trk_feats,
             "trk_feats": trk_feats,
             "labels": labels,
            }

with open(out_dir_data+"/preprocessed_"+dataset_tag+"_"+run_num+".pkl","wb") as f:
    pickle.dump(data_dict, f)

plt.figure()
plt.title("CosTheta")
plt.hist(costheta)
plt.savefig(out_dir_plots+"/run_"+run_num+"/costheta.png")

jet_feat_names = ["jet_pT","jet_eta","jet_phi","jet_m"]
jet_trk_feat_names = ["trk_pT","trk_eta","trk_phi","trk_q","trk_d0","trk_z0","trk_fromDown","trk_fromUp","trk_fromBottom"]
trk_feat_names = ["trk_pT","trk_eta","trk_phi","trk_q","trk_d0","trk_z0"]

for i, var in enumerate(jet_feat_names):
    feat=jet_feat_list[i]
    mini=ak.mean(feat)-2*ak.std(feat)
    maxi=ak.mean(feat)+2*ak.std(feat)
    plt.figure()
    plt.title(var)
    plt.hist(ak.ravel(feat),range=(mini,maxi),histtype='step',bins=50)
    plt.yscale('log')
    plt.savefig(out_dir_plots+"/run_"+run_num+"/"+var+".png")
    plt.close()

for i, var in enumerate(jet_trk_feat_names):
    feat=jet_trk_feat_list[i]
    mini=ak.mean(feat)-2*ak.std(feat)
    maxi=ak.mean(feat)+2*ak.std(feat)
    plt.figure()
    plt.title(var)
    plt.hist(ak.ravel(feat),range=(mini,maxi),histtype='step',bins=50)
    plt.yscale('log')
    plt.savefig(out_dir_plots+"/run_"+run_num+"/jet_"+var+".png")
    plt.close()

for i, var in enumerate(trk_feat_names):
    feat=trk_feat_list[i]
    mini=ak.mean(feat)-2*ak.std(feat)
    maxi=ak.mean(feat)+2*ak.std(feat)
    plt.figure()
    plt.title(var)
    plt.hist(ak.ravel(feat),range=(mini,maxi),histtype='step',bins=50)
    plt.yscale('log')
    plt.savefig(out_dir_plots+"/run_"+run_num+"/"+var+".png")
    plt.close()

print("Done!")
