import pickle
import awkward as ak
import numpy as np
import torch
import random
import sys

tag = str(sys.argv[1])
run = str(sys.argv[2])
out_dir_data= str(sys.argv[3])

with open(out_dir_data+"/run_"+run+"/preprocessed_"+tag+"_"+run+".pkl","rb") as f:
    data_dict = pickle.load( f )
jet_feats = data_dict["jet_feats"]
jet_trk_feats = data_dict["jet_trk_feats"]
trk_feats = data_dict["trk_feats"]
labels = data_dict["labels"]

#p = np.random.permutation(len(labels))
#jet_feats = jet_feats[p]
#jet_trk_feats = jet_trk_feats[p]
#trk_feats = trk_feats[p]
#labels = labels[p]

num_trks = ak.num(jet_trk_feats)
sort = ak.argsort(num_trks)

jet_feats = jet_feats[sort]
jet_trk_feats = jet_trk_feats[sort]
trk_feats = trk_feats[sort]
labels = labels[sort]

batch_size = 128
num_batches = int(len(labels)/batch_size)

num_jet_trk_feats=len(jet_trk_feats[0][0])-3
num_trk_feats=len(trk_feats[0][0])

jet_feats_batch = []
jet_trk_feats_batch = []
trk_feats_batch = []
labels_batch = []
jet_trk_labels_batch = []

for batch in range(num_batches):
    if batch%1==0:
        print("\tPadding Batch: ", batch+1, " / ", num_batches, end="\r")
    
    # Padding number of tracks per jet
    jet_trk_feat_list = []
    for feat in range(num_jet_trk_feats):
        batch_jet_trk_feats = jet_trk_feats[batch*batch_size:(batch+1)*batch_size,:,feat]        
        max_num_trks = ak.max(ak.num(batch_jet_trk_feats))
        pad_feat = ak.fill_none(ak.pad_none(batch_jet_trk_feats, max_num_trks, axis=1), 0)
        jet_trk_feat_list.append(pad_feat)

    jet_trk_feat_list = [x[:,:,np.newaxis] for x in jet_trk_feat_list]
    jet_trk_feats_combined = ak.concatenate(jet_trk_feat_list, axis=2)

    jet_trk_fromDown = jet_trk_feats[batch*batch_size:(batch+1)*batch_size,:,-3]
    jet_trk_fromUp = jet_trk_feats[batch*batch_size:(batch+1)*batch_size,:,-2]
    jet_trk_fromBottom = jet_trk_feats[batch*batch_size:(batch+1)*batch_size,:,-1]

    jet_trk_fromDown = ak.fill_none(ak.pad_none(jet_trk_fromDown, max_num_trks, axis=1), 0)
    jet_trk_fromUp = ak.fill_none(ak.pad_none(jet_trk_fromUp, max_num_trks, axis=1), 0)
    jet_trk_fromBottom = ak.fill_none(ak.pad_none(jet_trk_fromBottom, max_num_trks, axis=1), 0)

    jet_trk_labels = [jet_trk_fromDown[:,:,np.newaxis],jet_trk_fromUp[:,:,np.newaxis],jet_trk_fromBottom[:,:,np.newaxis]]
    jet_trk_labels_combined = ak.concatenate(jet_trk_labels, axis=2)

        
    # Padding number of tracks per event
    trk_feat_list = []
    for feat in range(num_trk_feats):
        batch_trk_feats = trk_feats[batch*batch_size:(batch+1)*batch_size,:,feat]        
        max_num_trks = ak.max(ak.num(batch_trk_feats))
        pad_feat = ak.fill_none(ak.pad_none(batch_trk_feats, max_num_trks, axis=1), 0)
        trk_feat_list.append(pad_feat)

    trk_feat_list = [x[:,:,np.newaxis] for x in trk_feat_list]
    trk_feats_combined = ak.concatenate(trk_feat_list, axis=2)

    jet_tensor = torch.tensor(jet_feats[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32)
    jet_trk_tensor = torch.tensor(jet_trk_feats_combined, dtype=torch.float32)
    trk_tensor = torch.tensor(trk_feats_combined, dtype=torch.float32)
    labels_tensor = torch.tensor(labels[batch*batch_size:(batch+1)*batch_size], dtype=torch.float32)
    jet_trk_labels_tensor = torch.tensor(jet_trk_labels_combined, dtype=torch.float32)
    #jet_trk_fromDown_tensor = torch.unsqueeze(torch.tensor(jet_trk_fromDown, dtype=torch.float32),2)
    #jet_trk_fromUp_tensor = torch.unsqueeze(torch.tensor(jet_trk_fromUp, dtype=torch.float32),2)
    #jet_trk_fromBottom_tensor = torch.unsqueeze(torch.tensor(jet_trk_fromBottom, dtype=torch.float32),2)
        
    jet_feats_batch.append(jet_tensor)
    jet_trk_feats_batch.append(jet_trk_tensor)
    trk_feats_batch.append(trk_tensor)
    labels_batch.append(labels_tensor)
    jet_trk_labels_batch.append(jet_trk_labels_tensor)

temp=list(zip(jet_feats_batch,jet_trk_feats_batch,trk_feats_batch,labels_batch,jet_trk_labels_batch))
random.shuffle(temp)
jet_feats_batch,jet_trk_feats_batch,trk_feats_batch,labels_batch,jet_trk_labels_batch=zip(*temp)

data_dict = {"jet_batch": jet_feats_batch,
             "jet_trk_batch": jet_trk_feats_batch,
             "trk_batch": trk_feats_batch,
             "label_batch": labels_batch,
             "jet_trk_labels_batch": jet_trk_labels_batch,
            }

with open(out_dir_data+"/run_"+run+"/data_batched_MSE_"+tag+"_"+run+".pkl","wb") as f:
    pickle.dump(data_dict, f)
