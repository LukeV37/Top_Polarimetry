import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model import Model, Stack, Encoder

def eval_model(tag, in_data, in_model, analysis_type):
    if analysis_type=="top":
        num_feats = 4
    if analysis_type=="bottom" or analysis_type=="down":
        num_feats = 3

    with open(in_data+"/data_batched_combined_MSE_"+tag+".pkl","rb") as f:
        data_dict = pickle.load( f )

    num_events = len(data_dict["label_batch"])

    X_test_jet = data_dict["jet_batch"]
    X_test_jet_trk = data_dict["jet_trk_batch"]
    X_test_trk = data_dict["trk_batch"]
    y_test = data_dict["label_batch"]
    y_test_jet_trk = data_dict["jet_trk_labels_batch"]

    num_batches=len(y_test)
    events_per_batch=len(y_test[0])
    num_events=num_batches*events_per_batch

    print("Evaluation Batches: ", num_batches)
    print("Evaluation Events: ", num_events)

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

        if analysis_type=="top":
            target = y_test[i][:,0:4]
        if analysis_type=="bottom":
            target = y_test[i][:,4:7]
        if analysis_type=="down":
            target = y_test[i][:,7:10]

        output, jet_trk_output = model(X_test_jet[i].to(device),
                       X_test_jet_trk[i].to(device),
                       X_test_trk[i].to(device),
                      )

        MSE_loss=MSE_loss_fn(output, target.to(device))
        CCE_jet_trk_loss=CCE_jet_trk_loss_fn(jet_trk_output, y_test_jet_trk[i].to(device))

        loss = 100*MSE_loss + CCE_jet_trk_loss
        #loss = MSE_loss

        cumulative_loss_test+=loss.detach().cpu().numpy().mean()

        predicted_labels = np.vstack((predicted_labels,output.detach().cpu().numpy()))
        if analysis_type=="top":
            true_labels = np.vstack((true_labels,target.detach().cpu().numpy()))
        if analysis_type=="bottom":
            true_labels = np.vstack((true_labels,target.detach().cpu().numpy()))
        if analysis_type=="down":
            true_labels = np.vstack((true_labels,target.detach().cpu().numpy()))

    cumulative_loss_test = cumulative_loss_test / num_test

    print("Test Loss:\t", cumulative_loss_test)
    print()
    print("Test MAE:\t", mean_absolute_error(true_labels, predicted_labels))
    print("Test RMSE:\t", root_mean_squared_error(true_labels, predicted_labels))

    return true_labels, predicted_labels

def eval_plots(title, true_labels, predicted_labels, analysis_type):
    if analysis_type=="bottom":
        feats = ['bottom_px','bottom_py','bottom_pz']
        ranges = [(-1,1),(-1,1),(-1,1)]
    if analysis_type=="down":
        feats = ['down_px','down_py','down_pz']
        ranges = [(-1,1),(-1,1),(-1,1)]
    if analysis_type=="top":
        feats = ['top_px','top_py','top_pz','top_E']
        ranges = [(-1000,1000),(-1000,1000),(-1000,1000),(0,1500)]
    num_feats=len(feats)

    print("Plotting predictions...")
    for i in range(num_feats):
        plt.figure()
        plt.hist(np.ravel(true_labels[:,i]),histtype='step',color='r',label='True Distribution',bins=50,range=ranges[i])
        plt.hist(np.ravel(predicted_labels[:,i]),histtype='step',color='b',label='Predicted Distribution',bins=50,range=ranges[i])
        plt.title("Predicted Ouput Distribution using Attention Model")
        plt.legend()
        plt.yscale('log')
        plt.xlabel(feats[i],loc='right')
        #plt.savefig(out_dir+"/pred_1d_"+feats[i]+".png")
        #plt.show()
        plt.close()

        plt.figure()
        plt.title("Ouput Distribution using Attention Model "+title)
        plt.hist2d(np.ravel(predicted_labels[:,i]),np.ravel(true_labels[:,i]), bins=100,norm=mcolors.LogNorm(),range=(ranges[i],ranges[i]))
        plt.xlabel('Predicted '+feats[i],loc='right')
        plt.ylabel('True '+feats[i],loc='top')
        diff = ranges[i][1] - ranges[i][0]
        plt.text(ranges[i][1]-0.3*diff,ranges[i][0]+0.2*diff,"$R^2$ value: "+str(round(r2_score(np.ravel(true_labels[:,i]),np.ravel(predicted_labels[:,i])),3)),backgroundcolor='r',color='k')
        #print("R^2 value: ", round(r2_score(predicted_labels[:,i],true_labels[:,i]),3))
        #plt.savefig(out_dir+"/pred_2d_"+feats[i]+".png")
        plt.show()
        plt.close()

    print("Done Evaluating!")