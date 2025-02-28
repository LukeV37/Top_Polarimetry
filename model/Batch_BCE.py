import pickle
import awkward as ak
import numpy as np
import torch
import random
import sys

tag1 = str(sys.argv[1])
tag2 = str(sys.argv[2])

with open("preprocessed_"+tag1+".pkl","rb") as f:
    data_dict_L = pickle.load( f )
jet_feats_L = data_dict_L["jet_feats"]
jet_trk_feats_L = data_dict_L["jet_trk_feats"]
trk_feats_L = data_dict_L["trk_feats"]
labels_L = ak.zeros_like(jet_feats_L[:,0])[:,np.newaxis]

with open("preprocessed_"+tag2+".pkl","rb") as f:
    data_dict_R = pickle.load( f )
jet_feats_R = data_dict_R["jet_feats"]
jet_trk_feats_R = data_dict_R["jet_trk_feats"]
trk_feats_R = data_dict_R["trk_feats"]
labels_R = ak.ones_like(jet_feats_R[:,0])[:,np.newaxis]

jet_feats = ak.concatenate([jet_feats_L,jet_feats_R],axis=0)
jet_trk_feats = ak.concatenate([jet_trk_feats_L,jet_trk_feats_R],axis=0)
trk_feats = ak.concatenate([trk_feats_L,trk_feats_R],axis=0)
labels = ak.concatenate([labels_L,labels_R],axis=0)

p = np.random.permutation(len(labels))

jet_feats = jet_feats[p]
jet_trk_feats = jet_trk_feats[p]
trk_feats = trk_feats[p]
labels = labels[p]

num_trks = ak.num(jet_trk_feats)
sort = ak.argsort(num_trks)

jet_feats = jet_feats[sort]
jet_trk_feats = jet_trk_feats[sort]
trk_feats = trk_feats[sort]
labels = labels[sort]

batch_size = 64
num_batches = int(len(labels)/batch_size)

num_feats=len(trk_feats[0][0])-1

jet_feats_batch = []
jet_trk_feats_batch = []
trk_feats_batch = []
labels_batch = []
jet_trk_labels_batch = []
trk_labels_batch = []

for batch in range(num_batches):
    if batch%1==0:
        print("\tPadding Batch: ", batch+1, " / ", num_batches, end="\r")
    
    jet_trk_feat_list = []
    for feat in range(num_feats):
        batch_jet_trk_feats = jet_trk_feats[batch*batch_size:(batch+1)*batch_size,:,feat]        
        max_num_trks = ak.max(ak.num(batch_jet_trk_feats))
        pad_feat = ak.fill_none(ak.pad_none(batch_jet_trk_feats, max_num_trks, axis=1), 0)
        jet_trk_feat_list.append(pad_feat)
        
    jet_trk_feat_list = [x[:,:,np.newaxis] for x in jet_trk_feat_list]
    jet_trk_feats_combined = ak.concatenate(jet_trk_feat_list, axis=2)

    jet_trk_labels = jet_trk_feats[batch*batch_size:(batch+1)*batch_size,:,-1]
    jet_trk_labels = ak.fill_none(ak.pad_none(jet_trk_labels, max_num_trks, axis=1), 0)
        
    trk_feat_list = []
    for feat in range(num_feats):
        batch_trk_feats = trk_feats[batch*batch_size:(batch+1)*batch_size,:,feat]        
        max_num_trks = ak.max(ak.num(batch_trk_feats))
        pad_feat = ak.fill_none(ak.pad_none(batch_trk_feats, max_num_trks, axis=1), 0)
        trk_feat_list.append(pad_feat)

    trk_feat_list = [x[:,:,np.newaxis] for x in trk_feat_list]
    trk_feats_combined = ak.concatenate(trk_feat_list, axis=2)

    trk_labels = trk_feats[batch*batch_size:(batch+1)*batch_size,:,-1]
    trk_labels = ak.fill_none(ak.pad_none(trk_labels, max_num_trks, axis=1), 0)
        
    jet_tensor = torch.tensor(jet_feats[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32)
    jet_trk_tensor = torch.tensor(jet_trk_feats_combined, dtype=torch.float32)
    trk_tensor = torch.tensor(trk_feats_combined, dtype=torch.float32)
    labels_tensor = torch.tensor(labels[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32)
    jet_trk_labels_tensor = torch.unsqueeze(torch.tensor(jet_trk_labels, dtype=torch.float32),2)
    trk_labels_tensor = torch.unsqueeze(torch.tensor(trk_labels, dtype=torch.float32),2)
        
    jet_feats_batch.append(jet_tensor)
    jet_trk_feats_batch.append(jet_trk_tensor)
    trk_feats_batch.append(trk_tensor)
    labels_batch.append(labels_tensor)
    jet_trk_labels_batch.append(jet_trk_labels_tensor)
    trk_labels_batch.append(trk_labels_tensor)

temp=list(zip(jet_feats_batch,jet_trk_feats_batch,trk_feats_batch,labels_batch,jet_trk_labels_batch,trk_labels_batch))
random.shuffle(temp)
jet_feats_batch,jet_trk_feats_batch,trk_feats_batch,labels_batch,jet_trk_labels_batch,trk_labels_batch=zip(*temp)

data_dict = {"jet_batch": jet_feats_batch,
             "jet_trk_batch": jet_trk_feats_batch,
             "trk_batch": trk_feats_batch,
             "label_batch": labels_batch,
             "jet_trk_label_batch": jet_trk_labels_batch,
             "trk_label_batch": trk_labels_batch,
            }

with open("data_batched_combined_"+tag1+"_"+tag2+".pkl","wb") as f:
    pickle.dump(data_dict, f)
