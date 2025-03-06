import pickle
import sys

tag1 = str(sys.argv[1])
tag2 = str(sys.argv[2])
num_runs = int(sys.argv[3])
out_dir_data= str(sys.argv[4])

combined_jet = []
combined_jet_trk = []
combined_trk = []
combined_label = []
combined_jet_trk_label = []

def get_data(out_dir_data,tag1,tag2,run):
    with open(out_dir_data+"/data_batched_"+tag1+"_"+tag2+"_"+str(run)+".pkl","rb") as f:
        data_dict = pickle.load(f)
    return data_dict["jet_batch"], data_dict["jet_trk_batch"], data_dict["trk_batch"], data_dict["label_batch"], data_dict["jet_trk_labels_batch"]

for i in range(num_runs):
    print("\tCombining Batch ", i)
    jet,jet_trk,trk,label,jet_trk_label = get_data(out_dir_data,tag1,tag2,i)
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

with open(out_dir_data+"/data_batched_combined_BCE_"+tag1+"_"+tag2+".pkl","wb") as f:
    pickle.dump(data_dict, f)
print("\tDone Combining Batches!")
