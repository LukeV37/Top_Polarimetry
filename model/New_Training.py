import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score
import sys
from new_model import *
from DataLoader_Parallel import CustomDataset

tag = str(sys.argv[1])
epochs = int(sys.argv[2])
embed_dim = int(sys.argv[3])
dir_dataset = str(sys.argv[4])
dir_training = str(sys.argv[5])

dir_startingPoint = "WS_U_10M/training_All_Task_LabFrame_Bottom_arctanh_80epoch_64embed"

starting_new = True
continue_training = not starting_new

# Loss parameters
alpha   = 1       # Top Loss
beta    = 0       # Down Loss
gamma   = 1000       # Direct Loss
delta   = 1000000       # Bottom Loss
epsilon = 0       # KL Loss

zeta = 0           # Track loss

batch_size=256
learning_rate=0.0001

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
if starting_new:
    model = Model(embed_dim,num_heads).to(device)
if continue_training:
    model = torch.load(dir_startingPoint+"/model_final.torch",weights_only=False,map_location=torch.device(device))

step_size=160
gamma=0.1
#optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

MSE_loss_fn = nn.MSELoss()
CCE_loss_fn = nn.CrossEntropyLoss()
kl_loss     = nn.KLDivLoss(reduction="batchmean")

print("Trainable Parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Number of Training Events: ", len(train_loader)*batch_size)

for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, direct_labels, track_labels in train_loader:
    top_pred, quark_pred, direct_pred, track_pred = model(probe_jet.to(device), constituents.to(device), event.to(device))
    break

def train(model, optimizer, train_loader, val_loader, epochs=40):
    
    combined_history = []
    
    for e in range(epochs):
        model.train()
        cumulative_loss_train = 0
        num_train = len(train_loader)

        for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, direct_labels, track_labels in train_loader:
            optimizer.zero_grad()
            
            top_pred, quark_pred, direct_pred, track_pred = model(probe_jet.to(device), constituents.to(device), event.to(device))

            bottom_labels = torch.atanh(bottom_labels * 0.999)
            bottom_costheta = torch.atanh(direct_labels[:,1].reshape(-1,1))

            top_loss      = MSE_loss_fn(top_pred, top_labels.to(device))
            down_loss     = MSE_loss_fn(quark_pred, down_labels.to(device))
            bottom_loss     = MSE_loss_fn(quark_pred, bottom_labels.to(device))
            costheta_loss = MSE_loss_fn(direct_pred, bottom_costheta.to(device))
            track_loss    = CCE_loss_fn(track_pred, track_labels.to(device))

            loss  = alpha*top_loss + beta*down_loss + gamma*costheta_loss + delta*bottom_loss + zeta*track_loss

            loss.backward()
            optimizer.step()
            
            cumulative_loss_train+=loss.detach().cpu().numpy().mean()
            
        cumulative_loss_train = cumulative_loss_train / num_train
        
        model.eval()
        cumulative_loss_val = 0
        cumulative_loss_top_val = 0
        cumulative_loss_down_val = 0
        cumulative_loss_bottom_val = 0
        cumulative_loss_direct_val = 0
        cumulative_loss_trk_val= 0
        num_val = len(val_loader)
        for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, direct_labels, track_labels in val_loader:
            top_pred, quark_pred, direct_pred, track_pred = model(probe_jet.to(device), constituents.to(device), event.to(device))

            bottom_labels = torch.atanh(bottom_labels * 0.999)
            bottom_costheta = torch.atanh(direct_labels[:,1].reshape(-1,1))

            top_loss      = MSE_loss_fn(top_pred, top_labels.to(device))
            down_loss     = MSE_loss_fn(quark_pred, down_labels.to(device))
            bottom_loss     = MSE_loss_fn(quark_pred, bottom_labels.to(device))
            costheta_loss = MSE_loss_fn(direct_pred, bottom_costheta.to(device))
            track_loss    = CCE_loss_fn(track_pred, track_labels.to(device))

            loss  = alpha*top_loss + beta*down_loss + gamma*costheta_loss + delta*bottom_loss + zeta*track_loss

            cumulative_loss_val+=loss.detach().cpu().numpy().mean()
            cumulative_loss_top_val+=top_loss.detach().cpu().numpy().mean()
            cumulative_loss_down_val+=down_loss.detach().cpu().numpy().mean()
            cumulative_loss_bottom_val+=bottom_loss.detach().cpu().numpy().mean()
            cumulative_loss_direct_val+=costheta_loss.detach().cpu().numpy().mean()
            cumulative_loss_trk_val+=track_loss.detach().cpu().numpy().mean()
        
        cumulative_loss_val = cumulative_loss_val / num_val
        cumulative_loss_top_val = alpha*cumulative_loss_top_val / num_val
        cumulative_loss_down_val = beta*cumulative_loss_down_val / num_val
        cumulative_loss_direct_val = gamma*cumulative_loss_direct_val / num_val
        cumulative_loss_bottom_val = delta*cumulative_loss_bottom_val / num_val
        cumulative_loss_trk_val= zeta*cumulative_loss_trk_val / num_val
        
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        scheduler.step()

        if e%1==0:
            print('Epoch:',e+1,'\tTrain Loss:',round(cumulative_loss_train,6),'\tVal Loss:',round(cumulative_loss_val,6))
            print('\t\t\t\t\tTop Loss: ', round(cumulative_loss_top_val,6))
            print('\t\t\t\t\tDown Loss: ', round(cumulative_loss_down_val,6))
            print('\t\t\t\t\tBottom Loss: ', round(cumulative_loss_bottom_val,6))
            print('\t\t\t\t\tDirect Loss: ', round(cumulative_loss_direct_val,6))
            print('\t\t\t\t\tTrack Loss: ', round(cumulative_loss_trk_val,6))

        torch.save(model,dir_training+"/models/model_Epoch_"+str(e+1)+".torch")
            
    return np.array(combined_history)

history = train(model, optimizer, train_loader, val_loader, epochs=epochs)

torch.save(model,dir_training+"/model_final.torch")

plt.figure()
plt.plot(history[:,0], label="Train")
plt.plot(history[:,1], label="Val")
plt.title('Loss')
plt.legend()
plt.yscale('log')
plt.savefig(dir_training+"/loss_curve_total.png")
#plt.show()

plt.figure()
plt.plot(history[int(epochs/2):,0], label="Train")
plt.plot(history[int(epochs/2):,1], label="Val")
plt.title('Loss')
plt.legend()
plt.yscale('log')
plt.savefig(dir_training+"/loss_curve_second_half.png")
#plt.show()

top_feats=4
pred_top = np.array([]).reshape(0,top_feats)
true_top = np.array([]).reshape(0,top_feats)

quark_feats=3
pred_quark = np.array([]).reshape(0,quark_feats)
true_quark = np.array([]).reshape(0,quark_feats)

direct_feats=1
pred_direct = np.array([]).reshape(0,direct_feats)
true_direct = np.array([]).reshape(0,direct_feats)

for probe_jet, constituents, event, top_labels, down_labels, bottom_labels, direct_labels, track_labels in test_loader:
    top_pred, quark_pred, direct_pred, track_pred = model(probe_jet.to(device), constituents.to(device), event.to(device))

    quark_pred = torch.tanh(quark_pred)
    direct_pred = torch.tanh(direct_pred)

    pred_top = np.vstack((pred_top,top_pred.detach().cpu().numpy()))
    true_top = np.vstack((true_top,top_labels.detach().cpu().numpy()))

    pred_quark = np.vstack((pred_quark,quark_pred.detach().cpu().numpy()))
    true_quark = np.vstack((true_quark,bottom_labels.detach().cpu().numpy()))

    pred_direct = np.vstack((pred_direct,direct_pred.detach().cpu().numpy()))
    true_direct = np.vstack((true_direct,direct_labels[:,1].reshape(-1,1).detach().cpu().numpy()))

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
                   "costheta": (-1.1,1.1)}

    for i ,var in enumerate(var_names):
        var_range = ranges_dict[var]

        plt.figure()
        plt.hist(np.ravel(true[:,i]),histtype='step',color='r',label='True Distribution',bins=50,range=var_range)
        plt.hist(np.ravel(pred[:,i]),histtype='step',color='b',label='Predicted Distribution',bins=50,range=var_range)
        plt.title("Predicted Ouput Distribution using Attention Model")
        plt.legend()
        plt.yscale('log')
        plt.xlabel(var_names[i],loc='right')
        plt.savefig(dir_training+"/pred_1d_"+var_names[i]+".png")
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
        plt.savefig(dir_training+"/pred_2d_"+var_names[i]+".png")
        #plt.show()
        plt.close()

validate_predictions(true_top, pred_top, ["top_px", "top_py", "top_pz", "top_e"])
#validate_predictions(true_down, pred_down, ["down_px", "down_py", "down_pz"])
validate_predictions(true_quark, pred_quark, ["bottom_px", "bottom_py", "bottom_pz"])
validate_predictions(true_direct, pred_direct, ["costheta"])
