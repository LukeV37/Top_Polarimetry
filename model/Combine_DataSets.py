import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import sys

from DataLoader_Parallel import CustomDataset

tag = str(sys.argv[1])
num_files = int(sys.argv[2])
dataset_dir = str(sys.argv[3])

file_list = [dataset_dir+'/run_'+str(i)+'/dataset.pt' for i in range(num_files)]

dataset_list = []

for file in file_list:
    dataset_list.append(torch.load(file, weights_only=False))

concat_dataset = ConcatDataset(dataset_list)

torch.save(concat_dataset, dataset_dir+"/dataset_combined.pt")
