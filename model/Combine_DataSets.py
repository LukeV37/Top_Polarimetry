
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import sys

tag = str(sys.argv[1])
num_files = int(sys.argv[2])
dataset_dir = str(sys.argv[3])

file_list = [dataset_dir+'/run_'+str(i)+'/dataset.pt' for i in range(num_files)]

class CustomDataset(Dataset):
    def __init__(self):
        self.lepton = 0
        self.nu = 0
        self.probe_jet = 0
        self.probe_jet_constituents = 0
        self.balance_jets = 0
        self.labels = 0
        self.track_labels = 0
    def __getitem__(self, idx):
        return self.lepton[idx], self.nu[idx], self.probe_jet[idx], self.probe_jet_constituents[idx], self.balance_jets[idx], self.labels[idx], self.track_labels[idx]
    def __len__(self):
        return len(self.lepton)

dataset_list = []

for file in file_list:
    dataset_list.append(torch.load(file, weights_only=False))

concat_dataset = ConcatDataset(dataset_list)

train_dataset, test_dataset = torch.utils.data.random_split(concat_dataset, [0.75, 0.25])
val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [0.2, 0.8])

torch.save(train_dataset, dataset_dir+"/train_dataset.pt")
torch.save(val_dataset, dataset_dir+"/val_dataset.pt")
torch.save(test_dataset, dataset_dir+"/test_dataset.pt")
