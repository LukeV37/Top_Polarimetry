import pickle
import sys

tag = str(sys.argv[1])
num_runs = int(sys.argv[2])
out_dir_data= str(sys.argv[3])

combined_jet = []
combined_jet_trk = []
combined_trk = []
combined_label = []
combined_jet_trk_label = []

def get_data(out_dir_data,tag,run):
    with open(out_dir_data+"/run_"+run+"/data_batched_MSE_"+tag+"_"+run+".pkl","rb") as f:
        data_dict = pickle.load(f)
    return data_dict["jet_batch"], data_dict["jet_trk_batch"], data_dict["trk_batch"], data_dict["label_batch"], data_dict["jet_trk_labels_batch"]

for i in range(num_runs):
    print("\tCombining Batch ", i)
    jet,jet_trk,trk,label,jet_trk_label = get_data(out_dir_data,tag,str(i))
    combined_jet+=jet
    combined_jet_trk+=jet_trk
    combined_trk+=trk
    combined_label+=label
    combined_jet_trk_label+=jet_trk_label

data_dict = {
    "jet_batch": combined_jet,
    "jet_trk_batch": combined_jet_trk,
    "trk_batch": combined_trk,
    "label_batch": combined_label,
    "jet_trk_labels_batch": combined_jet_trk_label,
}

with open(out_dir_data+"/data_batched_combined_MSE_"+tag+".pkl","wb") as f:
    pickle.dump(data_dict, f)
print("\tDone Combining Batches!")
