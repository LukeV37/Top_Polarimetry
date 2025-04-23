import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score
import sys

from model import Model

tag = str(sys.argv[1])
Epochs = int(sys.argv[2])
Step = int(sys.argv[3])
in_dir = str(sys.argv[4])
out_dir = str(sys.argv[5])
analysis_type = str(sys.argv[6])

assert analysis_type=="bottom" or analysis_type=="down" or analysis_type=="top"

if analysis_type=="top":
    feats = ['top_px','top_py','top_pz','top_E']
if analysis_type=="bottom":
    feats = ['bottom_px','bottom_py','bottom_pz']
if analysis_type=="down":
    feats = ['down_px','down_py','down_pz']
num_feats = len(feats)

with open(in_dir+"/data_batched_combined_MSE_"+tag+".pkl","rb") as f:
    data_dict = pickle.load( f )

num_events = len(data_dict["label_batch"])

train_split = int(0.7 * num_events)
test_split = int(0.75 * num_events)

X_train_jet = data_dict["jet_batch"][0:train_split]
X_train_jet_trk = data_dict["jet_trk_batch"][0:train_split]
X_train_trk = data_dict["trk_batch"][0:train_split]
y_train = data_dict["label_batch"][0:train_split]
y_train_jet_trk = data_dict["jet_trk_labels_batch"][0:train_split]

X_val_jet = data_dict["jet_batch"][train_split:test_split]
X_val_jet_trk = data_dict["jet_trk_batch"][train_split:test_split]
X_val_trk = data_dict["trk_batch"][train_split:test_split]
y_val = data_dict["label_batch"][train_split:test_split]
y_val_jet_trk = data_dict["jet_trk_labels_batch"][train_split:test_split]

X_test_jet = data_dict["jet_batch"][test_split:]
X_test_jet_trk = data_dict["jet_trk_batch"][test_split:]
X_test_trk = data_dict["trk_batch"][test_split:]
y_test = data_dict["label_batch"][test_split:]
y_test_jet_trk = data_dict["jet_trk_labels_batch"][test_split:]

print("Training Batches: ", len(y_train))
print("Validation Batches: ", len(y_val))
print("Testing Batches: ", len(y_test))

def NormLoss(pred):
    # pred is shape (N,3) where N is batch size
    norm = torch.square(pred[:,0])+torch.square(pred[:,1])+torch.square(pred[:,2])
    loss = torch.square(norm-1) # Expecatation value is 1 so remove mean
    return torch.mean(loss) # Use mean as reduction operation

### Define Training Loop
def train(X_train_jet, X_train_jet_trk, X_train_trk, y_train, y_train_jet_trk,
          X_val_jet, X_val_jet_trk, X_val_trk, y_val, y_val_jet_trk,
          epochs=40):
    
    combined_history = []
    
    num_train = len(X_train_jet)
    num_val = len(X_val_jet)
    
    step_size=Step
    gamma=0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for e in range(epochs):
        model.train()
        cumulative_loss_train = 0

        for i in range(num_train):
            #print("Current:", torch.cuda.memory_allocated()/1e9)
            #print("Max:    ",torch.cuda.max_memory_reserved()/1e9)
            #print()
            optimizer.zero_grad()
            
            output, jet_trk_output = model(X_train_jet[i].to(device),
                                                       X_train_jet_trk[i].to(device),
                                                       X_train_trk[i].to(device),
                                                      )
            
            MSE_loss=MSE_loss_fn(output, y_train[i].to(device))
            CCE_jet_trk_loss=CCE_jet_trk_loss_fn(jet_trk_output, y_train_jet_trk[i].to(device))
            Norm_loss=NormLoss(output)
            
            if analysis_type=="bottom" or analysis_type=="down":
                loss = 100*MSE_loss + CCE_jet_trk_loss + Norm_loss
            if analysis_type=="top":
                loss = 100*MSE_loss + CCE_jet_trk_loss
            #loss = MSE_loss

            loss.backward()
            optimizer.step()
                        
            cumulative_loss_train+=loss.detach().cpu().numpy().mean()

        cumulative_loss_train = cumulative_loss_train / num_train
        
        model.eval()
        cumulative_loss_val = 0
        for i in range(num_val):
            output, jet_trk_output = model(X_val_jet[i].to(device),
                                                       X_val_jet_trk[i].to(device),
                                                       X_val_trk[i].to(device),
                                                      )
            
            MSE_loss=MSE_loss_fn(output, y_val[i].to(device))
            CCE_jet_trk_loss=CCE_jet_trk_loss_fn(jet_trk_output, y_val_jet_trk[i].to(device))
            Norm_loss=NormLoss(output)

            if analysis_type=="bottom" or analysis_type=="down":
                loss = 100*MSE_loss + CCE_jet_trk_loss + Norm_loss
            if analysis_type=="top":
                loss = 100*MSE_loss + CCE_jet_trk_loss
            #loss = MSE_loss

            cumulative_loss_val+=loss.detach().cpu().numpy().mean()

        scheduler.step()
            
        cumulative_loss_val = cumulative_loss_val / num_val
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        if e%1==0:
            print('Epoch:',e+1,'\tTrain Loss:',round(cumulative_loss_train,6),'\tVal Loss:',round(cumulative_loss_val,6))
 
        if (e+1)%step_size==0:
            print("\tReducing Step Size by ", gamma)

        torch.save(model,out_dir+"/model_Epoch_"+str(e+1)+".torch")
            
    return np.array(combined_history)

print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print()

model = Model(64,4,num_feats,analysis_type)
model.to(device)
print("Trainable Parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.AdamW(model.parameters(), lr=0.001)
MSE_loss_fn = nn.MSELoss()
CCE_jet_trk_loss_fn = nn.CrossEntropyLoss()

combined_history = train(X_train_jet, X_train_jet_trk, X_train_trk, y_train, y_train_jet_trk,
                         X_val_jet, X_val_jet_trk, X_val_trk, y_val, y_val_jet_trk,
                         epochs=Epochs)
torch.save(model,out_dir+"/model_final.torch")

torch.cuda.empty_cache()

plt.figure()
plt.plot(combined_history[:,0], label="Train")
plt.plot(combined_history[:,1], label="Val")
plt.title('Loss')
plt.legend()
plt.yscale('log')
plt.savefig(out_dir+"/Loss_Curve.png")
#plt.show()

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
    Norm_loss=NormLoss(output)

    if analysis_type=="bottom" or analysis_type=="down":
        loss = 100*MSE_loss + CCE_jet_trk_loss + Norm_loss
    if analysis_type=="top":
        loss = 100*MSE_loss + CCE_jet_trk_loss
    #loss = MSE_loss

    cumulative_loss_test+=loss.detach().cpu().numpy().mean()

    predicted_labels = np.vstack((predicted_labels,output.detach().cpu().numpy()))
    true_labels = np.vstack((true_labels,y_test[i].detach().cpu().numpy()))
    
cumulative_loss_test = cumulative_loss_test / num_test
    
print("Train Loss:\t", combined_history[-1][0])
print("Val Loss:\t", combined_history[-1][1])
print("Test Loss:\t", cumulative_loss_test)
print()
print("Test MAE:\t", mean_absolute_error(true_labels, predicted_labels))
print("Test RMSE:\t", root_mean_squared_error(true_labels, predicted_labels))

#feats = ['top_px','top_py','top_pz','top_E','down_px','down_py','down_pz','down_pT','down_eta','down_phi','down_deltaR','down_deltaEta','down_deltaPhi','bottom_px','bottom_py','bottom_pz','bottom_pT','bottom_eta','bottom_phi','bottom_deltaR','bottom_deltaEta','bottom_deltaPhi','costheta']
#ranges = [(-1000,1000),(-1000,1000),(-1000,1000),(0,1500),(-1,1),(-1,1),(-1,1),(0,600),(-5,5),(-3.14,3.14),(0,3),(-2,2),(-3,3),(-1,1),(-1,1),(-1,1),(0,600),(-5,5),(-3.14,3.14),(0,3),(-2,2),(-3,3),(-1,1)]
if analysis_type=="down" or analysis_type=="bottom":
    ranges = [(-1,1),(-1,1),(-1,1)]
if analysis_type=="top":
    ranges = [(-1000,1000),(-1000,1000),(-1000,1000),(0,1500)]

print("Plotting predictions...")
for i in range(num_feats):
    plt.figure()
    plt.hist(np.ravel(true_labels[:,i]),histtype='step',color='r',label='True Distribution',bins=50,range=ranges[i])
    plt.hist(np.ravel(predicted_labels[:,i]),histtype='step',color='b',label='Predicted Distribution',bins=50,range=ranges[i])
    plt.title("Predicted Ouput Distribution using Attention Model")
    plt.legend()
    plt.yscale('log')
    plt.xlabel(feats[i],loc='right')
    plt.savefig(out_dir+"/pred_1d_"+feats[i]+".png")
    #plt.show()
    plt.close()

    plt.figure()
    plt.title("Ouput Distribution using Attention Model")
    plt.hist2d(np.ravel(predicted_labels[:,i]),np.ravel(true_labels[:,i]), bins=100,norm=mcolors.LogNorm(),range=(ranges[i],ranges[i]))
    plt.xlabel('Predicted '+feats[i],loc='right')
    plt.ylabel('True '+feats[i],loc='top')
    diff = ranges[i][1] - ranges[i][0]
    plt.text(ranges[i][1]-0.3*diff,ranges[i][0]+0.2*diff,"$R^2$ value: "+str(round(r2_score(np.ravel(predicted_labels[:,i]),np.ravel(true_labels[:,i])),3)),backgroundcolor='r',color='k')
    #print("R^2 value: ", round(r2_score(predicted_labels[:,i],true_labels[:,i]),3))
    plt.savefig(out_dir+"/pred_2d_"+feats[i]+".png")
    #plt.show()
    plt.close()

print("Done training!")
