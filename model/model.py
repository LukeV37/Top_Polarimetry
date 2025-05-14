import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def forward(self, jet_embedding, jet_trk_embedding, trk_embedding):
        # Jet Track Attention
        jet_trk_embedding, jet_trk_weights = self.jet_trk_encoder(jet_trk_embedding, jet_trk_embedding, jet_trk_embedding)
        # Cross Attention (Local)
        jet_embedding, cross_weights = self.jet_trk_cross_encoder(jet_embedding, jet_trk_embedding, jet_trk_embedding)
        # Track Attention
        trk_embedding, trk_weights = self.trk_encoder(trk_embedding, trk_embedding, trk_embedding)
        # Cross Attention (Global)
        jet_embedding, cross_weights = self.trk_cross_encoder(jet_embedding, trk_embedding, trk_embedding)
        return jet_embedding, jet_trk_embedding, trk_embedding

class Model(nn.Module):  
    def __init__(self,embed_dim, num_heads, num_labels, analysis_type):
        super(Model, self).__init__()   
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_labels = num_labels
        self.analysis_type = analysis_type
        self.num_jet_feats = 4
        self.num_trk_feats = 6
        
        # Initiliazer
        self.jet_initializer = nn.Linear(self.num_jet_feats, self.embed_dim)
        self.jet_trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)
        self.trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)
           
        # Transformer Stack
        self.stack1 = Stack(self.embed_dim, self.num_heads)
        self.stack2 = Stack(self.embed_dim, self.num_heads)
        self.stack3 = Stack(self.embed_dim, self.num_heads)

        # Regression Task
        self.regression = nn.Linear(self.embed_dim, self.num_labels)
        # Classification Task
        self.jet_trk_classification = nn.Linear(self.embed_dim, 3)

    def forward(self, jets, jet_trks, trks):
        
        # Feature preprocessing layers
        jet_embedding = F.gelu(self.jet_initializer(jets))
        jet_trk_embedding = F.gelu(self.jet_trk_initializer(jet_trks))
        trk_embedding = F.gelu(self.trk_initializer(trks))
        jet_embedding = torch.unsqueeze(jet_embedding, 1)
        
        # Transformer Stack
        jet_embedding, jet_trk_embedding, trk_embedding = self.stack1(jet_embedding,jet_trk_embedding,trk_embedding)
        jet_embedding, jet_trk_embedding, trk_embedding = self.stack2(jet_embedding,jet_trk_embedding,trk_embedding)
        jet_embedding, jet_trk_embedding, trk_embedding = self.stack3(jet_embedding,jet_trk_embedding,trk_embedding)
    
        # Get output
        jet_embedding = torch.squeeze(jet_embedding,1)
        if self.analysis_type=="bottom" or self.analysis_type=="down":
            output = F.tanh(self.regression(jet_embedding))
        if self.analysis_type=="top":
            output = self.regression(jet_embedding)
        if self.analysis_type=="direct":
            output = 2*F.tanh(self.regression(jet_embedding))
        jet_trk_classification = self.jet_trk_classification(jet_trk_embedding)
        
        return output, jet_trk_classification
