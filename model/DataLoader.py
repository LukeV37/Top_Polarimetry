import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/11201
import pickle
import os
import time

file_list = ["data_batched_MSE_U_7M_"+str(i)+".pkl" for i in range(10)]

def load_file(file):
    print(file)
    with open(file, 'rb') as f:
        data_dict = pickle.load(f)
    jet_feats = data_dict["jet_batch"]
    jet_trk_feats = data_dict["jet_trk_batch"]
    trk_feats = data_dict["trk_batch"]
    jet_labels = data_dict["label_batch"]
    subjet_labels = data_dict["jet_trk_labels_batch"]
    
    print(jet_feats[0])
    
    return jet_feats, jet_trk_feats, trk_feats, jet_labels, subjet_labels

class CustomDataset(Dataset):
    def __init__(self, path, files):
        self.data_files = [path+file for file in files]
    
    def __getitem__(self, idx):
        return load_file(self.data_files[idx])     

    def __len__(self):
        return len(self.data_files)

dset = CustomDataset('./WS_U_7M/datasets/',file_list)
loader = DataLoader(dset, num_workers=4)

for jet_feats, jet_trk_feats, trk_feats, jet_labels, subjet_labels in loader:
    print(jet_feats[0])
