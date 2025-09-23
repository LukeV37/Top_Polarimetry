import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/11201
import sys

tag = str(sys.argv[1])
file_num = int(sys.argv[2])
dataset_dir = str(sys.argv[3])

file = '../pythia/WS_'+tag+'/dataset_selected_'+tag+'_'+str(file_num)+'.root:fastjet'

def sort_by_pT(object_feature_dict):
    pT = object_feature_dict["pT"]
    sorted_idx = ak.argsort(pT, ascending=False)
    sorted_feat_dict ={}
    for key in object_feature_dict:
        sorted_feat_dict[key] = object_feature_dict[key][sorted_idx]
    return sorted_feat_dict

def clip_to_num(object_feature_dict, clip_num):
    clipped_feat_dict = {}
    for key in object_feature_dict:
        clipped_feat_dict[key] = ak.fill_none(ak.pad_none(object_feature_dict[key], clip_num, clip=True), 0)
    return clipped_feat_dict

def combine_feats(feat_list, axis):
    if axis == 1:
        feat_list = [feat[:,np.newaxis] for feat in feat_list]
        combined_feats = ak.concatenate(feat_list, axis=axis)
    if axis == 2:
        feat_list = [feat[:,:,np.newaxis] for feat in feat_list]
        combined_feats = ak.concatenate(feat_list, axis=axis)
    combined_feats = torch.tensor(combined_feats, dtype=torch.float32)
    return combined_feats

def get_norm(px, py, pz):
    norm = np.sqrt(px**2+py**2+pz**2)
    return px/norm, py/norm, pz/norm

def load_file(file):
    print(file)
    with uproot.open(file) as f:
        # Input Features
        lepton_pT = f["lepton_pT"].array()
        lepton_eta = f["lepton_eta"].array()
        lepton_phi = f["lepton_phi"].array()
        lepton_q = f["lepton_q"].array()
        nu_MET = f["nu_MET"].array()
        nu_phi = f["nu_phi"].array()
        probe_jet_pT = f["probe_jet_pT"].array()
        probe_jet_eta = f["probe_jet_eta"].array()
        probe_jet_phi = f["probe_jet_phi"].array()
        probe_jet_mass = f["probe_jet_mass"].array()
        probe_jet_constituent_pT= f["probe_jet_constituent_pT"].array()
        probe_jet_constituent_eta= f["probe_jet_constituent_eta"].array()
        probe_jet_constituent_phi= f["probe_jet_constituent_phi"].array()
        probe_jet_constituent_q= f["probe_jet_constituent_q"].array()
        probe_jet_constituent_PID= f["probe_jet_constituent_PID"].array()
        balance_jets_pT = f["balance_jets_pT"].array()
        balance_jets_eta = f["balance_jets_eta"].array()
        balance_jets_phi = f["balance_jets_phi"].array()

        # Output Labels
        truth_top_px_boosted = f["truth_top_px_boosted"].array()
        truth_top_py_boosted = f["truth_top_py_boosted"].array()
        truth_top_pz_boosted = f["truth_top_pz_boosted"].array()
        truth_top_e_boosted = f["truth_top_e_boosted"].array()
        truth_down_px_boosted = f["truth_down_px_boosted"].array()
        truth_down_py_boosted = f["truth_down_py_boosted"].array()
        truth_down_pz_boosted = f["truth_down_pz_boosted"].array()
        #truth_down_e_boosted = f["truth_down_e_boosted"].array()
        truth_costheta = f["costheta"].array()

        # Track Labels
        probe_jet_constituent_fromDown= f["probe_jet_constituent_fromDown"].array()
        probe_jet_constituent_fromUp= f["probe_jet_constituent_fromUp"].array()
        probe_jet_constituent_fromBottom= f["probe_jet_constituent_fromBottom"].array()

    # Combine features into single tensor
    lepton_feats = combine_feats([lepton_pT, lepton_eta, lepton_phi, lepton_q], axis=1)
    nu_feats = combine_feats([nu_MET, nu_phi], axis=1)
    probe_jet_feats = combine_feats([probe_jet_pT, probe_jet_eta, probe_jet_phi, probe_jet_mass], axis=1)

    # Combine feats for probe jet constituents: sort by pT, clip to max num, combine feats
    max_constituent_num=200
    probe_jet_constituent_var_list = ["pT", "eta", "phi", "q", "PID", "fromDown", "fromUp", "fromBottom"]
    probe_jet_constituent_dict = {"pT": probe_jet_constituent_pT, "eta": probe_jet_constituent_eta, "phi": probe_jet_constituent_phi, "q": probe_jet_constituent_q, "PID": probe_jet_constituent_PID,
                                  "fromDown": probe_jet_constituent_fromDown, "fromUp": probe_jet_constituent_fromUp, "fromBottom": probe_jet_constituent_fromBottom}
    sorted_probe_jet_constituent_dict = sort_by_pT(probe_jet_constituent_dict)
    clipped_probe_jet_constituent_dict = clip_to_num(sorted_probe_jet_constituent_dict, max_constituent_num)
    probe_jet_constituent_feats = combine_feats([clipped_probe_jet_constituent_dict["pT"], clipped_probe_jet_constituent_dict["eta"], clipped_probe_jet_constituent_dict["phi"], clipped_probe_jet_constituent_dict["q"], clipped_probe_jet_constituent_dict["PID"]], axis=2)

    # Combine feats for balance jets: sort by pT, clip to max num, combine feats
    max_balance_jet_num=10
    balance_jet_var_list = ["pT", "eta", "phi"]
    balance_jet_dict = {"pT": balance_jets_pT, "eta": balance_jets_eta, "phi": balance_jets_phi}
    sorted_balanced_jet_dict = sort_by_pT(balance_jet_dict)
    clipped_balance_jet_dict = clip_to_num(sorted_balanced_jet_dict, max_balance_jet_num)
    balance_jets_feats = combine_feats([clipped_balance_jet_dict["pT"], clipped_balance_jet_dict["eta"], clipped_balance_jet_dict["phi"]], axis=2)

    # Combine labels into single tensor
    truth_down_px_norm, truth_down_py_norm, truth_down_pz_norm = get_norm(truth_down_px_boosted, truth_down_py_boosted, truth_down_pz_boosted)
    top_labels = combine_feats([truth_top_px_boosted, truth_top_py_boosted, truth_top_pz_boosted, truth_top_e_boosted], axis=1)
    down_labels = combine_feats([truth_down_px_norm, truth_down_py_norm, truth_down_pz_norm], axis=1)
    direct_labels = combine_feats([truth_costheta], axis=1)
    track_labels = combine_feats([clipped_probe_jet_constituent_dict["fromDown"], clipped_probe_jet_constituent_dict["fromUp"], clipped_probe_jet_constituent_dict["fromBottom"]], axis=2)
    
    return lepton_feats, nu_feats, probe_jet_feats, probe_jet_constituent_feats, balance_jets_feats, top_labels, down_labels, direct_labels, track_labels

class CustomDataset(Dataset):
    def __init__(self, file):

        lepton_feats, nu_feats, probe_jet_feats, probe_jet_constituent_feats, balance_jets_feats, top_labels, down_labels, direct_labels, track_labels = load_file(file)

        self.lepton = lepton_feats
        self.nu = nu_feats
        self.probe_jet = probe_jet_feats
        self.probe_jet_constituents = probe_jet_constituent_feats
        self.balance_jets = balance_jets_feats
        self.top_labels = top_labels
        self.down_labels = down_labels
        self.direct_labels = direct_labels
        self.track_labels = track_labels
    
    def __getitem__(self, idx):
        return self.lepton[idx], self.nu[idx], self.probe_jet[idx], self.probe_jet_constituents[idx], self.balance_jets[idx], self.top_labels[idx], self.down_labels[idx], self.direct_labels[idx], self.track_labels[idx]

    def __len__(self):
        return len(self.lepton)

dset = CustomDataset(file)
torch.save(dset, dataset_dir+"/run_"+str(file_num)+"/dataset.pt")
