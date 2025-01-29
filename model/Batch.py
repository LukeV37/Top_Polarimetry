import pickle
import awkward as ak
import numpy as np
import torch
with open("data.pkl","rb") as f:
    data_dict = pickle.load( f )

jet_feats = data_dict["jet_feats"]
trk_feats = data_dict["trk_feats"]
labels = data_dict["label"]

num_trks = ak.num(trk_feats)
sort = ak.argsort(num_trks)

jet_feats = jet_feats[sort]
trk_feats = trk_feats[sort]
labels = labels[sort]

batch_size = 128
num_batches = int(len(labels)/batch_size)

num_feats=len(trk_feats[0][0])

jet_feats_batch = []
trk_feats_batch = []
labels_batch = []

for batch in range(num_batches):
    if batch%1==0:
        print("\tPadding Batch: ", batch+1, " / ", num_batches, end="\r")
    
    trk_feat_list = []
    for feat in range(num_feats):
        batch_trk_feats = trk_feats[batch*batch_size:(batch+1)*batch_size,:,feat]
        max_num_trks = ak.max(ak.num(batch_trk_feats))
        pad = ak.fill_none(ak.pad_none(batch_trk_feats, max_num_trks, axis=1), 0)
        trk_feat_list.append(pad)
        
    trk_feat_list = [x[:,:,np.newaxis] for x in trk_feat_list]
    trk_feats_combined = ak.concatenate(trk_feat_list, axis=2)
    
    jet_tensor = torch.tensor(jet_feats[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32)
    trk_tensor = torch.tensor(trk_feats_combined, dtype=torch.float32)
    labels_tensor = torch.tensor(labels[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32)
        
    jet_feats_batch.append(jet_tensor)
    trk_feats_batch.append(trk_tensor)
    labels_batch.append(labels_tensor)

data_dict = {"jet_batch": jet_feats_batch,
             "trk_batch": trk_feats_batch,
             "label_batch": labels_batch,
            }

with open("data_batched.pkl","wb") as f:
    pickle.dump(data_dict, f)
