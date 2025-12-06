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
        return latent

class Stack(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Stack, self).__init__()
        self.jet_trk_encoder = Encoder(embed_dim, num_heads)
        self.trk_encoder = Encoder(embed_dim, num_heads)
        self.jet_trk_cross_encoder = Encoder(embed_dim, num_heads)
        self.trk_cross_encoder = Encoder(embed_dim, num_heads)
    def forward(self, jet_embedding, jet_trk_embedding, trk_embedding):
        # Jet Track Attention
        jet_trk_embedding = self.jet_trk_encoder(jet_trk_embedding, jet_trk_embedding, jet_trk_embedding)
        # Cross Attention (Local)
        jet_embedding = self.jet_trk_cross_encoder(jet_embedding, jet_trk_embedding, jet_trk_embedding)
        # Track Attention
        trk_embedding = self.trk_encoder(trk_embedding, trk_embedding, trk_embedding)
        # Cross Attention (Global)
        jet_embedding = self.trk_cross_encoder(jet_embedding, trk_embedding, trk_embedding)
        return jet_embedding, jet_trk_embedding, trk_embedding

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initiliazer
        self.probe_jet_initializer = nn.Linear(4, self.embed_dim)
        self.probe_jet_constituent_initializer = nn.Linear(4, self.embed_dim)
        self.event_initializer = nn.Linear(4, self.embed_dim)
           
        # Transformer Stack
        self.stack1 = Stack(self.embed_dim, self.num_heads)
        self.stack2 = Stack(self.embed_dim, self.num_heads)
        self.stackTop = Stack(self.embed_dim, self.num_heads)
        self.stackQuark1 = Stack(self.embed_dim, self.num_heads)
        self.stackQuark2 = Stack(self.embed_dim, self.num_heads)

        # Track Classification
        self.track_classification = nn.Linear(self.embed_dim, 3)
        
        # Kinematics Regression
        self.top_regression = nn.Linear(self.embed_dim, 4)
        self.quark_regression = nn.Linear(self.embed_dim, 3)

        # Direct regression
        self.direct_input = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.direct_output = nn.Linear(self.embed_dim, 2)

    def forward(self, probe_jet, probe_jet_constituent, event_tensor):
        
        # Feature initialization layers
        probe_jet_embedding = F.gelu(self.probe_jet_initializer(probe_jet))
        probe_jet_constituent_embedding = F.gelu(self.probe_jet_constituent_initializer(probe_jet_constituent))
        event_embedding = F.gelu(self.event_initializer(event_tensor))
        
        # Transformer Encoder Stack
        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stack1(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding = probe_jet_embedding + probe_jet_embedding_NEW
        probe_jet_constituent_embedding = probe_jet_constituent_embedding + probe_jet_constituent_embedding_NEW
        event_embedding = event_embedding + event_embedding_NEW

        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stack2(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding = probe_jet_embedding + probe_jet_embedding_NEW
        probe_jet_constituent_embedding = probe_jet_constituent_embedding + probe_jet_constituent_embedding_NEW
        event_embedding = event_embedding + event_embedding_NEW

        # Track Classificiation
        track_output = self.track_classification(probe_jet_constituent_embedding)
        
        # Top Encoder Stack
        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stackTop(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding_Top = probe_jet_embedding + probe_jet_embedding_NEW

        # Down Encoder Stack
        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stackQuark1(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding = probe_jet_embedding + probe_jet_embedding_NEW
        probe_jet_constituent_embedding = probe_jet_constituent_embedding + probe_jet_constituent_embedding_NEW
        event_embedding = event_embedding + event_embedding_NEW
        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stackQuark2(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding_Quark = probe_jet_embedding + probe_jet_embedding_NEW

        # Contract first dimension
        probe_jet_embedding_Top  = torch.squeeze(probe_jet_embedding_Top,1)
        probe_jet_embedding_Quark = torch.squeeze(probe_jet_embedding_Quark,1)
        
        # Get Top output
        top_output = self.top_regression(probe_jet_embedding_Top)
        
        # Get Down output
        quark_output = self.quark_regression(probe_jet_embedding_Quark)

        # Get Direct output
        combined_output = torch.cat([probe_jet_embedding_Top,probe_jet_embedding_Quark], axis=1)
        costheta_output = F.gelu(self.direct_input(combined_output))
        costheta_output = self.direct_output(costheta_output)
        
        return top_output, quark_output, costheta_output, track_output
