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

tag = str(sys.argv[1])
epochs = int(sys.argv[2])
embed_dim = int(sys.argv[3])
dir_dataset = str(sys.argv[4])
dir_training = str(sys.argv[5])

class CustomDataset(Dataset):
    def __init__(self):
        self.lepton = 0
        self.nu = 0
        self.probe_jet = 0
        self.probe_jet_constituents = 0
        self.balance_jets = 0
        self.top_labels = 0
        self.down_labels = 0
        self.direct_labels = 0
        self.track_labels = 0
    def __getitem__(self, idx):
        return self.lepton[idx], self.nu[idx], self.probe_jet[idx], self.probe_jet_constituents[idx], self.balance_jets[idx], self.top_labels[idx], self.down_labels[idx], self.direct_labels[idx], self.track_labels[idx]
    def __len__(self):
        return len(self.lepton)

batch_size=128

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

model = Model2(embed_dim,4).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0001)

MSE_loss_fn = nn.MSELoss()
CCE_loss_fn = nn.CrossEntropyLoss()

print("Trainable Parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Number of Training Events: ", len(train_loader)*batch_size)

for lepton, MET, probe_jet, constituents, small_jet, top_labels, down_labels, direct_labels, track_labels in train_loader:
    top_pred, down_pred, direct_pred, trk_output = model(lepton.to(device), MET.to(device), probe_jet.to(device), constituents.to(device), small_jet.to(device))
    break

def train(model, optimizer, train_loader, val_loader, epochs=40):
    
    combined_history = []
    
    for e in range(epochs):
        model.train()
        cumulative_loss_train = 0
        num_train = len(train_loader)

        for lepton, MET, probe_jet, constituents, small_jet, top_labels, down_labels, direct_labels, track_labels in train_loader:
            optimizer.zero_grad()
            
            top_pred, down_pred, direct_pred, trk_output = model(lepton.to(device), MET.to(device), probe_jet.to(device), constituents.to(device), small_jet.to(device))
                        
            top_loss      = MSE_loss_fn(top_pred, top_labels.to(device))
            down_loss     = MSE_loss_fn(down_pred, down_labels.to(device))
            costheta_loss = MSE_loss_fn(direct_pred, direct_labels.to(device))
            track_loss    = CCE_loss_fn(trk_output, track_labels.to(device))
            
            alpha = 0
            beta  = 1
            gamma = 0
            delta = 0
            loss  = alpha*top_loss + beta*down_loss + gamma*costheta_loss + delta*track_loss
            
            loss.backward()
            optimizer.step()
            
            cumulative_loss_train+=loss.detach().cpu().numpy().mean()
            
        cumulative_loss_train = cumulative_loss_train / num_train
        
        model.eval()
        cumulative_loss_val = 0
        num_val = len(val_loader)
        for lepton, MET, probe_jet, constituents, small_jet, top_labels, down_labels, direct_labels, track_labels in val_loader:
            top_pred, down_pred, direct_pred, trk_output = model(lepton.to(device), MET.to(device), probe_jet.to(device), constituents.to(device), small_jet.to(device))
            
            top_loss      = MSE_loss_fn(top_pred, top_labels.to(device))
            down_loss     = MSE_loss_fn(down_pred, down_labels.to(device))
            costheta_loss = MSE_loss_fn(direct_pred, direct_labels.to(device))
            track_loss    = CCE_loss_fn(trk_output, track_labels.to(device))
            
            loss  = alpha*top_loss + beta*down_loss + gamma*costheta_loss + delta*track_loss

            cumulative_loss_val+=loss.detach().cpu().numpy().mean()
        
        cumulative_loss_val = cumulative_loss_val / num_val
        
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        if e%1==0:
            print('Epoch:',e+1,'\tTrain Loss:',round(cumulative_loss_train,6),'\tVal Loss:',round(cumulative_loss_val,6))

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

down_feats=3
pred_down = np.array([]).reshape(0,down_feats)
true_down = np.array([]).reshape(0,down_feats)

direct_feats=1
pred_costheta = np.array([]).reshape(0,direct_feats)
true_costheta = np.array([]).reshape(0,direct_feats)

for lepton, MET, probe_jet, constituents, small_jet, top_labels, down_labels, direct_labels, track_labels in test_loader:
    top_pred, down_pred, direct_pred, trk_output = model(lepton.to(device), MET.to(device), probe_jet.to(device), constituents.to(device), small_jet.to(device))
    
    pred_top = np.vstack((pred_top,top_pred.detach().cpu().numpy()))
    true_top = np.vstack((true_top,top_labels.detach().cpu().numpy()))
    
    pred_down = np.vstack((pred_down,down_pred.detach().cpu().numpy()))
    true_down = np.vstack((true_down,down_labels.detach().cpu().numpy()))
    
    pred_costheta = np.vstack((pred_costheta,direct_pred.detach().cpu().numpy()))
    true_costheta = np.vstack((true_costheta,direct_labels.detach().cpu().numpy()))

do_PlotTop=False
do_PlotDown=True
do_PlotDirect=False

def validate_predictions(true, pred, var_names):
    num_feats = len(var_names)
    ranges_dict = {"top_px": (-1000,1000),
                   "top_py": (-1000,1000),
                   "top_pz": (-1000,1000),
                   "top_e" : (0,1500),
                   "down_px": (-1.2,1.2),
                   "down_py": (-1.2,1.2),
                   "down_pz": (-1.2,1.2),
                   "costheta": (-1.2,1.2)}

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

if do_PlotTop:
    validate_predictions(true_top, pred_top, ["top_px", "top_py", "top_pz", "top_e"])

if do_PlotDown:
    validate_predictions(true_down, pred_down, ["down_px", "down_py", "down_pz"])

if do_PlotDirect:
    validate_predictions(true_costheta, pred_costheta, ["costheta"])
