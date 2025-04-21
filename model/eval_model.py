import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score

from model import Model, Stack, Encoder

def eval_model(tag, in_data, in_model, out_dir, analysis_type):
    if analysis_type=="bottom":
        feats = ['bottom_px','bottom_py','bottom_pz']
    if analysis_type=="down":
        feats = ['down_px','down_py','down_pz']
    if analysis_type=="top":
        feats = ['top_px','top_py','top_pz','top_E']
    num_feats = len(feats)

    with open(in_data+"/data_batched_combined_MSE_"+tag+".pkl","rb") as f:
        data_dict = pickle.load( f )

    num_events = len(data_dict["label_batch"])

    train_split = int(0.7 * num_events)
    test_split = int(0.75 * num_events)

    X_test_jet = data_dict["jet_batch"]
    X_test_jet_trk = data_dict["jet_trk_batch"]
    X_test_trk = data_dict["trk_batch"]
    y_test = data_dict["label_batch"]
    y_test_jet_trk = data_dict["jet_trk_labels_batch"]

    print("Testing Batches: ", len(y_test))

    print("GPU Available: ", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print()

    model = torch.load(in_model+"/model_final.torch",weights_only=False,map_location=torch.device(device))

    MSE_loss_fn = nn.MSELoss()
    CCE_jet_trk_loss_fn = nn.CrossEntropyLoss()

    ### Evaluate Model
    model.eval()
    cumulative_loss_test = 0
    predicted_labels = []
    true_labels = []
    binary_pred = []
    binary_true = []

    predicted_labels = np.array([]).reshape(0,num_feats)
    true_labels = np.array([]).reshape(0,num_feats)

    num_test = len(X_test_jet)
    for i in range(num_test):
        output, jet_trk_output = model(X_test_jet[i].to(device),
                       X_test_jet_trk[i].to(device),
                       X_test_trk[i].to(device),
                      )

        MSE_loss=MSE_loss_fn(output, y_test[i].to(device))
        CCE_jet_trk_loss=CCE_jet_trk_loss_fn(jet_trk_output, y_test_jet_trk[i].to(device))

        loss = 100*MSE_loss + CCE_jet_trk_loss
        #loss = MSE_loss

        cumulative_loss_test+=loss.detach().cpu().numpy().mean()

        predicted_labels = np.vstack((predicted_labels,output.detach().cpu().numpy()))
        true_labels = np.vstack((true_labels,y_test[i].detach().cpu().numpy()))

    cumulative_loss_test = cumulative_loss_test / num_test

    print("Test Loss:\t", cumulative_loss_test)
    print()
    print("Test MAE:\t", mean_absolute_error(true_labels, predicted_labels))
    print("Test RMSE:\t", root_mean_squared_error(true_labels, predicted_labels))

    return true_labels, predicted_labels
