import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score
import sys
from new_model import *
from DataLoader_Parallel import CustomDataset

model_path = "WS_U_10M/training_Direct_Task_Boost_tRest_Bottom_cosSim_10M_60epoch_128embed/models/model_Epoch_60.torch"
dir_training="WS_U_10M/training_Direct_Task_Boost_tRest_Bottom_cosSim_10M_60epoch_128embed"
dir_dataset="WS_U_10M/datasets_AllFrame"

batch_size=256

dset = torch.load(dir_dataset+"/dataset_combined.pt", weights_only=False)

generator = torch.Generator().manual_seed(42)

train_dataset, test_dataset = torch.utils.data.random_split(dset, [0.75, 0.25], generator=generator)
val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [0.2, 0.8], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_heads=4
model = torch.load(model_path,weights_only=False,map_location=torch.device(device))

def calc_norm(y_pred):
    norm = torch.sqrt(torch.sum(torch.square(y_pred), dim=1)).reshape(-1,1)
    y_pred_norm = torch.div(y_pred, norm)
    return y_pred_norm

def uniform_to_circle(cos_theta):
    angle = torch.acos(cos_theta)
    return torch.cat([cos_theta, torch.sin(angle)], dim=1)

top_feats=4
pred_top = np.array([]).reshape(0,top_feats)
true_top = np.array([]).reshape(0,top_feats)

quark_feats=3
pred_quark = np.array([]).reshape(0,quark_feats)
true_quark = np.array([]).reshape(0,quark_feats)

direct_feats=1
cos_direct_pred = np.array([]).reshape(0,direct_feats)
sin_direct_pred = np.array([]).reshape(0,direct_feats)
cos_direct_true = np.array([]).reshape(0,direct_feats)
sin_direct_true = np.array([]).reshape(0,direct_feats)

for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, direct_labels, track_labels in test_loader:
    top_pred, quark_pred, direct_pred, track_pred = model(probe_jet.to(device), constituents.to(device), event.to(device))

    #quark_pred = calc_norm(quark_pred)
    direct_pred = calc_norm(direct_pred)
    direct_true = uniform_to_circle(direct_labels)

    #pred_top = np.vstack((pred_top,top_pred.detach().cpu().numpy()))
    #true_top = np.vstack((true_top,top_labels.detach().cpu().numpy()))

    #pred_quark = np.vstack((pred_quark,quark_pred.detach().cpu().numpy()))
    #true_quark = np.vstack((true_quark,bottom_labels.detach().cpu().numpy()))

    cos_direct_pred = np.vstack((cos_direct_pred,direct_pred[:,0].reshape(-1,1).detach().cpu().numpy()))
    sin_direct_pred = np.vstack((sin_direct_pred,direct_pred[:,1].reshape(-1,1).detach().cpu().numpy()))
    cos_direct_true = np.vstack((cos_direct_true,direct_true[:,0].reshape(-1,1).detach().cpu().numpy()))
    sin_direct_true = np.vstack((sin_direct_true,direct_true[:,1].reshape(-1,1).detach().cpu().numpy()))

def validate_predictions(true, pred, var_names):
    num_feats = len(var_names)
    ranges_dict = {"top_px": (-1000,1000),
                   "top_py": (-1000,1000),
                   "top_pz": (-1000,1000),
                   "top_e" : (0,1500),
                   "down_px": (-1.1,1.1),
                   "down_py": (-1.1,1.1),
                   "down_pz": (-1.1,1.1),
                   "bottom_px": (-1.1,1.1),
                   "bottom_py": (-1.1,1.1),
                   "bottom_pz": (-1.1,1.1),
                   "costheta": (-1.1,1.1),
                   "sintheta": (-0.1,1.1),
                   }

    for i ,var in enumerate(var_names):
        var_range = ranges_dict[var]

        plt.figure()
        plt.hist(np.ravel(true[:,i]),histtype='step',color='r',label='True Distribution',bins=50,range=var_range)
        plt.hist(np.ravel(pred[:,i]),histtype='step',color='b',label='Predicted Distribution',bins=50,range=var_range)
        plt.title("Predicted Ouput Distribution using Attention Model")
        plt.legend()
        plt.yscale('log')
        plt.xlabel(var_names[i],loc='right')
        plt.savefig(dir_training+"/pred_1d_"+var_names[i]+"_EVAL.png")
        #plt.show()
        plt.close()

        #plt.figure()
        fig, ax = plt.subplots()
        plt.title("Ouput Distribution using Attention Model")
        h = ax.hist2d(np.ravel(pred[:,i]),np.ravel(true[:,i]), bins=100,norm=mcolors.LogNorm(),range=(var_range,var_range))
        #fig.colorbar(h[3], ax=ax)
        plt.xlabel('Predicted '+var_names[i],loc='right')
        plt.ylabel('True '+var_names[i],loc='top')
        diff = var_range[1] - var_range[0]
        plt.text(var_range[1]-0.3*diff,var_range[0]+0.2*diff,"$R^2$ value: "+str(round(r2_score(np.ravel(true[:,i]),np.ravel(pred[:,i])),3)),backgroundcolor='r',color='k')
        #print("R^2 value: ", round(r2_score(true_labels[:,i],predicted_labels[:,i]),3))
        plt.savefig(dir_training+"/pred_2d_"+var_names[i]+"_EVAL.png")
        #plt.show()
        plt.close()

#validate_predictions(true_top, pred_top, ["top_px", "top_py", "top_pz", "top_e"])
#validate_predictions(true_down, pred_down, ["down_px", "down_py", "down_pz"])
#validate_predictions(true_quark, pred_quark, ["bottom_px", "bottom_py", "bottom_pz"])
validate_predictions(cos_direct_true, cos_direct_pred, ["costheta"])
validate_predictions(sin_direct_true, sin_direct_pred, ["sintheta"])
