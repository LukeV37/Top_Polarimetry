import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/11201
import sys

class CustomDataset(Dataset):
    def __init__(self, file):

        probe_jet_feats, probe_jet_constituent_feats, event_tensor_feats, top_labels, down_labels, bottom_labels, direct_labels, track_labels = load_file(file)

        self.probe_jet = probe_jet_feats
        self.probe_jet_constituents = probe_jet_constituent_feats
        self.event_tensor_feats = event_tensor_feats
        self.top_labels = top_labels
        self.down_labels = down_labels
        self.bottom_labels = bottom_labels
        self.direct_labels = direct_labels
        self.track_labels = track_labels
    
    def __getitem__(self, idx):
        return self.probe_jet[idx], self.probe_jet_constituents[idx], self.event_tensor_feats[idx], self.top_labels[idx], self.down_labels[idx], self.bottom_labels[idx],self.direct_labels[idx], self.track_labels[idx]

    def __len__(self):
        return len(self.probe_jet)

def sort_by_pT(object_feature_dict):
    pT = object_feature_dict["pT"]
    sorted_idx = ak.argsort(pT, ascending=False)
    sorted_feat_dict ={}
    for key in object_feature_dict:
        sorted_feat_dict[key] = object_feature_dict[key][sorted_idx]
    return sorted_feat_dict

def clip_to_num(object_feature_dict, clip_num, axis):
    clipped_feat_dict = {}
    for key in object_feature_dict:
        clipped_feat_dict[key] = ak.fill_none(ak.pad_none(object_feature_dict[key], clip_num, axis=axis, clip=True), 0)
    return clipped_feat_dict

def cut_neutrals(object_feature_dict):
    q = object_feature_dict["q"]
    mask = q!=0
    for key in object_feature_dict:
        object_feature_dict[key] = object_feature_dict[key][mask]
    return object_feature_dict

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
        top_px = f["top_px_boost_ttCM"].array()
        top_py = f["top_py_boost_ttCM"].array()
        top_pz = f["top_pz_boost_ttCM"].array()
        top_e  = f["top_e_boost_ttCM"].array()
        down_px = f["down_px_boost_tRest"].array()
        down_py = f["down_py_boost_tRest"].array()
        down_pz = f["down_pz_boost_tRest"].array()
        bottom_px = f["bottom_px_boost_tRest"].array()
        bottom_py = f["bottom_py_boost_tRest"].array()
        bottom_pz = f["bottom_pz_boost_tRest"].array()
        costheta_down = f["costheta_down"].array()
        costheta_bottom = f["costheta_bottom"].array()

        # Track Labels
        probe_jet_constituent_fromDown= f["probe_jet_constituent_fromDown"].array()
        probe_jet_constituent_fromUp= f["probe_jet_constituent_fromUp"].array()
        probe_jet_constituent_fromBottom= f["probe_jet_constituent_fromBottom"].array()

    # Clipping parameters
    max_constituent_num=200
    max_balance_jet_num=10

    # Initlize feature dictionaries
    probe_jet_dict = {"pT": probe_jet_pT, "eta": probe_jet_eta, "phi": probe_jet_phi, "mass": probe_jet_mass}
    probe_jet_constituent_dict = {"pT": probe_jet_constituent_pT, "eta": probe_jet_constituent_eta, "phi": probe_jet_constituent_phi, "q": probe_jet_constituent_q, "PID": probe_jet_constituent_PID,
                                  "fromDown": probe_jet_constituent_fromDown, "fromUp": probe_jet_constituent_fromUp, "fromBottom": probe_jet_constituent_fromBottom}
    lepton_dict = {"pT": lepton_pT, "eta": lepton_eta, "phi": lepton_phi, "q": lepton_q}
    MET_dict = {"MET": nu_MET, "phi": nu_phi}
    balance_jet_dict = {"pT": balance_jets_pT, "eta": balance_jets_eta, "phi": balance_jets_phi}

    # probe jet constituents: sort by pT, clip to max num, combine feats
    probe_jet_constituents_no_neutrals_dict = cut_neutrals(probe_jet_constituent_dict)
    sorted_probe_jet_constituent_dict = sort_by_pT(probe_jet_constituents_no_neutrals_dict)
    clipped_probe_jet_constituent_dict = clip_to_num(sorted_probe_jet_constituent_dict, max_constituent_num, axis=1)

    # Combine feats for balance jets: sort by pT, clip to max num, combine feats
    sorted_balanced_jet_dict = sort_by_pT(balance_jet_dict)
    clipped_balance_jet_dict = clip_to_num(sorted_balanced_jet_dict, max_balance_jet_num, axis=1)

    # Combine Feats
    probe_jet_feats = combine_feats([probe_jet_dict["pT"],probe_jet_dict["eta"],probe_jet_dict["phi"],probe_jet_dict["mass"]], axis=1)
    probe_jet_constituent_feats = combine_feats([clipped_probe_jet_constituent_dict["pT"], clipped_probe_jet_constituent_dict["eta"], clipped_probe_jet_constituent_dict["phi"], clipped_probe_jet_constituent_dict["q"]], axis=2)
    lepton_feats = combine_feats([lepton_dict["pT"],lepton_dict["eta"],lepton_dict["phi"],lepton_dict["q"]], axis=1)
    MET_feats = combine_feats([MET_dict["MET"], MET_dict["phi"]], axis=1)
    balance_jets_feats = combine_feats([clipped_balance_jet_dict["pT"], clipped_balance_jet_dict["eta"], clipped_balance_jet_dict["phi"]], axis=2)

    # Pad Feats to common length
    common_feat_len = 4
    probe_jet_feats = ak.fill_none(ak.pad_none(probe_jet_feats, common_feat_len, axis=1), 0)
    probe_jet_constituent_feats = ak.fill_none(ak.pad_none(probe_jet_constituent_feats, common_feat_len, axis=2), 0)
    lepton_feats = ak.fill_none(ak.pad_none(lepton_feats, common_feat_len, axis=1), 0)
    MET_feats = ak.fill_none(ak.pad_none(MET_feats, common_feat_len, axis=1), 0)
    balance_jets_feats = ak.fill_none(ak.pad_none(balance_jets_feats, common_feat_len, axis=2), 0)

    # Combine labels into single tensor
    down_px_norm, down_py_norm, down_pz_norm = get_norm(down_px, down_py, down_pz)
    bottom_px_norm, bottom_py_norm, bottom_pz_norm = get_norm(bottom_px, bottom_py, bottom_pz)
    top_labels = combine_feats([top_px, top_py, top_pz, top_e], axis=1)
    down_labels = combine_feats([down_px_norm, down_py_norm, down_pz_norm], axis=1)
    bottom_labels = combine_feats([bottom_px_norm, bottom_py_norm, bottom_pz_norm], axis=1)
    direct_labels = combine_feats([costheta_down, costheta_bottom], axis=1)
    track_labels = combine_feats([clipped_probe_jet_constituent_dict["fromDown"], clipped_probe_jet_constituent_dict["fromUp"], clipped_probe_jet_constituent_dict["fromBottom"]], axis=2)

    # Convert to tensors
    probe_jet_feats = torch.unsqueeze(torch.tensor(probe_jet_feats, dtype=torch.float32),1)
    probe_jet_constituent_feats = torch.tensor(probe_jet_constituent_feats, dtype=torch.float32)
    lepton_feats = torch.unsqueeze(torch.tensor(lepton_feats, dtype=torch.float32),1)
    MET_feats = torch.unsqueeze(torch.tensor(MET_feats, dtype=torch.float32),1)
    balance_jets_feats = torch.tensor(balance_jets_feats, dtype=torch.float32)

    # Construct event tensor
    event_tensor_feats = torch.cat([probe_jet_feats, probe_jet_constituent_feats, lepton_feats, MET_feats, balance_jets_feats], dim=1)

    return probe_jet_feats, probe_jet_constituent_feats, event_tensor_feats, top_labels, down_labels, bottom_labels, direct_labels, track_labels

if __name__=="__main__":
    tag = str(sys.argv[1])
    file_num = int(sys.argv[2])
    dataset_dir = str(sys.argv[3])

    file = '../pythia/WS_'+tag+'/dataset_selected_'+tag+'_'+str(file_num)+'.root:fastjet'

    dset = CustomDataset(file)
    torch.save(dset, dataset_dir+"/run_"+str(file_num)+"/dataset.pt")
