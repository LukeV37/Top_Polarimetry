import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score

tag = "down_500k_charm_500k"

with open("data_batched_combined_"+tag+".pkl","rb") as f:
    data_dict = pickle.load( f )

num_events = len(data_dict["label_batch"])

train_split = int(0.7 * num_events)
test_split = int(0.75 * num_events)

X_train_jet = data_dict["jet_batch"][0:train_split]
X_train_jet_trk = data_dict["jet_trk_batch"][0:train_split]
X_train_trk = data_dict["trk_batch"][0:train_split]
y_train = data_dict["label_batch"][0:train_split]
y_train_jet_trk = data_dict["jet_trk_label_batch"][0:train_split]
y_train_trk = data_dict["trk_label_batch"][0:train_split]

X_val_jet = data_dict["jet_batch"][train_split:test_split]
X_val_jet_trk = data_dict["jet_trk_batch"][train_split:test_split]
X_val_trk = data_dict["trk_batch"][train_split:test_split]
y_val = data_dict["label_batch"][train_split:test_split]
y_val_jet_trk = data_dict["jet_trk_label_batch"][train_split:test_split]
y_val_trk = data_dict["trk_label_batch"][train_split:test_split]

X_test_jet = data_dict["jet_batch"][test_split:]
X_test_jet_trk = data_dict["jet_trk_batch"][test_split:]
X_test_trk = data_dict["trk_batch"][test_split:]
y_test = data_dict["label_batch"][test_split:]
y_test_jet_trk = data_dict["jet_trk_label_batch"][test_split:]
y_test_trk = data_dict["trk_label_batch"][test_split:]

print("Training Batches: ", len(y_train))
print("Validation Batches: ", len(y_val))
print("Testing Batches: ", len(y_test))

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Encoder, self).__init__()
        self.pre_norm_Q = nn.LayerNorm(embed_dim)
        self.pre_norm_K = nn.LayerNorm(embed_dim)
        self.pre_norm_V = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads=num_heads,batch_first=True, dropout=0.25)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim,embed_dim)
    def forward(self, Query, Key, Value):
        Query = self.pre_norm_Q(Query)
        Key = self.pre_norm_K(Key)
        Value = self.pre_norm_V(Value)
        context, weights = self.attention(Query, Key, Value)
        context = self.post_norm(context)
        latent = Query + context
        tmp = F.gelu(self.out(latent))
        latent = latent + tmp
        return latent, weights

class Stack(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Stack, self).__init__()
        self.jet_trk_encoder = Encoder(embed_dim, num_heads)
        self.trk_encoder = Encoder(embed_dim, num_heads)
        self.jet_trk_cross_encoder = Encoder(embed_dim, num_heads)
        self.trk_cross_encoder = Encoder(embed_dim, num_heads)
        self.postprocess = nn.Linear(2*embed_dim, embed_dim)
    def forward(self, jet_embedding, jet_trk_embedding, trk_embedding):
        # Jet Track Attention
        jet_trk_embedding, jet_trk_weights = self.jet_trk_encoder(jet_trk_embedding, jet_trk_embedding, jet_trk_embedding)
        # Track Aggregation
        #jet_trk_aggregated = torch.unsqueeze(jet_trk_embedding.sum(dim=1),1)
        #jet_embedding = torch.cat((jet_embedding,jet_trk_aggregated),2)
        #jet_embedding = F.relu(self.postprocess(jet_embedding))
        # Cross Attention (Local)
        jet_embedding, cross_weights = self.jet_trk_cross_encoder(jet_embedding, jet_trk_embedding, jet_trk_embedding)
        # Track Attention
        #trk_embedding, trk_weights = self.trk_encoder(trk_embedding, trk_embedding, trk_embedding)
        # Cross Attention (Global)
        #jet_embedding, cross_weights = self.trk_cross_encoder(jet_embedding, trk_embedding, trk_embedding)
        return jet_embedding, jet_trk_embedding, trk_embedding

class Model(nn.Module):  
    def __init__(self):
        super(Model, self).__init__()   
        
        self.embed_dim = 32
        self.num_heads = 4
        self.num_jet_feats = 4
        self.num_trk_feats = 6
        
        # Initliazer
        self.jet_initializer = nn.Linear(self.num_jet_feats, self.embed_dim)
        self.jet_trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)
        self.trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)

        # Transformer stack
        self.stack1 = Stack(self.embed_dim, self.num_heads)
        self.stack2 = Stack(self.embed_dim, self.num_heads)
        #self.stack3 = Stack(self.embed_dim, self.num_heads)
        #self.stack4 = Stack(self.embed_dim, self.num_heads)
        #self.stack5 = Stack(self.embed_dim, self.num_heads)
        #self.stack6 = Stack(self.embed_dim, self.num_heads)
           
        # Classification Task
        self.classification = nn.Linear(self.embed_dim, 1)
        self.jet_trk_classification = nn.Linear(self.embed_dim, 1)
        self.trk_classification = nn.Linear(self.embed_dim, 1)
        
    def forward(self, jets, jet_trks, trks):
        
        # Feature preprocessing layers
        jet_embedding = F.gelu(self.jet_initializer(jets))
        jet_trk_embedding = F.gelu(self.jet_trk_initializer(jet_trks))
        trk_embedding = F.gelu(self.trk_initializer(trks))
        jet_embedding = torch.unsqueeze(jet_embedding, 1)

        # Transformer Stack
        jet_embedding, jet_trk_embedding, trk_embedding = self.stack1(jet_embedding,jet_trk_embedding,trk_embedding)
        jet_embedding, jet_trk_embedding, trk_embedding = self.stack2(jet_embedding,jet_trk_embedding,trk_embedding)
        #jet_embedding, jet_trk_embedding, trk_embedding = self.stack3(jet_embedding,jet_trk_embedding,trk_embedding)
        #jet_embedding, jet_trk_embedding, trk_embedding = self.stack4(jet_embedding,jet_trk_embedding,trk_embedding)
        #jet_embedding, jet_trk_embedding, trk_embedding = self.stack5(jet_embedding,jet_trk_embedding,trk_embedding)
        #jet_embedding, jet_trk_embedding, trk_embedding = self.stack6(jet_embedding,jet_trk_embedding,trk_embedding)
        
        # Get output
        jet_embedding = torch.squeeze(jet_embedding,1)
        output = F.sigmoid(self.classification(jet_embedding))
        jet_trk_classification = F.sigmoid(self.jet_trk_classification(jet_trk_embedding))
        trk_classification = F.sigmoid(self.trk_classification(trk_embedding))
        
        return output, jet_trk_classification, trk_classification

# ### Define Training Loop
def train(X_train_jet, X_train_jet_trk, X_train_trk, y_train, y_train_jet_trk, y_train_trk, 
          X_val_jet, X_val_jet_trk, X_val_trk, y_val, y_val_jet_trk, y_val_trk, 
          epochs=40):
    
    combined_history = []
    
    num_train = len(X_train_jet)
    num_val = len(X_val_jet)
    
    step_size=200
    gamma=0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for e in range(epochs):
        model.train()
        cumulative_loss_train = 0

        for i in range(num_train):
            optimizer.zero_grad()

            output, jet_trk_output, trk_output = model(X_train_jet[i].to(device),
                                                       X_train_jet_trk[i].to(device),
                                                       X_train_trk[i].to(device),
                                                      )

            jet_loss=loss_fn(output, y_train[i].to(device))
            BCE_jet_trk_loss=loss_fn(jet_trk_output, y_train_jet_trk[i].to(device))
            BCE_trk_loss=loss_fn(trk_output, y_train_trk[i].to(device))

            #loss = jet_loss + BCE_jet_trk_loss + BCE_trk_loss
            loss = jet_loss + BCE_jet_trk_loss
            #loss = jet_loss
            
            loss.backward()
            optimizer.step()
                        
            cumulative_loss_train+=loss.detach().cpu().numpy().mean()
            
        cumulative_loss_train = cumulative_loss_train / num_train

        torch.cuda.empty_cache()
        
        model.eval()
        cumulative_loss_val = 0
        for i in range(num_val):
            output, jet_trk_output, trk_output = model(X_val_jet[i].to(device),
                                                       X_val_jet_trk[i].to(device),
                                                       X_val_trk[i].to(device),
                                                      )

            jet_loss=loss_fn(output, y_val[i].to(device))
            BCE_jet_trk_loss=loss_fn(jet_trk_output, y_val_jet_trk[i].to(device))
            BCE_trk_loss=loss_fn(trk_output, y_val_trk[i].to(device))

            #loss = jet_loss + BCE_jet_trk_loss + BCE_trk_loss
            loss = jet_loss + BCE_jet_trk_loss
            #loss = jet_loss
            
            cumulative_loss_val+=loss.detach().cpu().numpy().mean()
            
            
        scheduler.step()
            
        cumulative_loss_val = cumulative_loss_val / num_val
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        if e%1==0:
            print('Epoch:',e+1,'\tTrain Loss:',round(cumulative_loss_train,6),'\tVal Loss:',round(cumulative_loss_val,6))
 
        if (e+1)%step_size==0:
            print("\tReducing Step Size by ", gamma)
            
        torch.cuda.empty_cache()
            
    return np.array(combined_history)

print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print()

model = Model()
model.to(device)
print("Trainable Parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))

Epochs=100
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
loss_fn = nn.BCELoss()

combined_history = train(X_train_jet, X_train_jet_trk, X_train_trk, y_train, y_train_jet_trk, y_train_trk,
                         X_val_jet, X_val_jet_trk, X_val_trk, y_val, y_val_jet_trk, y_val_trk,
                         epochs=Epochs)
torch.save(model,"model.torch")

plt.figure()
plt.plot(combined_history[:,0], label="Train")
plt.plot(combined_history[:,1], label="Val")
plt.title('Loss')
plt.legend()
plt.yscale('log')
plt.savefig("plots/Loss_Curve.png")
#plt.show()

### Evaluate Model
model.eval()
cumulative_loss_test = 0
predicted_labels = []
true_labels = []
binary_pred = []
binary_true = []

predicted_labels = np.array([]).reshape(0,1)
true_labels = np.array([]).reshape(0,1)

torch.cuda.empty_cache()

num_test = len(X_test_jet)
for i in range(num_test):
    output, jet_trk_output, trk_output = model(X_test_jet[i].to(device),
                   X_test_jet_trk[i].to(device),
                   X_test_trk[i].to(device),
                  )

    torch.cuda.empty_cache()

    jet_loss=loss_fn(output, y_test[i].to(device))
    BCE_jet_trk_loss=loss_fn(jet_trk_output, y_test_jet_trk[i].to(device))
    BCE_trk_loss=loss_fn(trk_output, y_test_trk[i].to(device))

    #loss = jet_loss + BCE_jet_trk_loss + BCE_trk_loss
    loss = jet_loss + BCE_jet_trk_loss
    #loss = jet_loss

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

L = true_labels==0
R = true_labels==1

plt.figure()
#plt.hist(true_labels,histtype='step',color='r',label='True Distribution',bins=50,range=(0,1))
plt.hist(predicted_labels[L],histtype='step',color='b',label='Down-Type',bins=30,range=(0,1))
plt.hist(predicted_labels[R],histtype='step',color='r',label='Charm-Type',bins=30,range=(0,1))

plt.title("Predicted Ouput Distribution using Attention Model")
plt.legend()
#plt.yscale('log')
plt.xlabel('isCharm',loc='right')
plt.text(0.7,70,"ROC AUC: "+str(round(roc_auc_score(true_labels, predicted_labels),3)))
plt.savefig("plots/pred_BCE.png")
#plt.show()
