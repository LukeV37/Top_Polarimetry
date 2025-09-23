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
        return_weights=False
        if return_weights:
            return latent,weights
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

class Model2(nn.Module):  
    def __init__(self, embed_dim, num_heads):
        super(Model2, self).__init__()   
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initiliazer
        self.lepton_initializer = nn.Linear(4, self.embed_dim)
        self.MET_initializer = nn.Linear(2, self.embed_dim)
        self.probe_jet_initializer = nn.Linear(4, self.embed_dim)
        self.probe_jet_constituent_initializer = nn.Linear(5, self.embed_dim)
        self.small_jet_initializer = nn.Linear(3, self.embed_dim)
           
        # Transformer Stack
        self.stack1 = Stack(self.embed_dim, self.num_heads)
        self.stack2 = Stack(self.embed_dim, self.num_heads)
        #self.stackTop = Stack(self.embed_dim, self.num_heads)
        self.stackDown= Stack(self.embed_dim, self.num_heads)
        
        # Kinematics Regression
        #self.top_regression_input = nn.Linear(self.embed_dim, self.embed_dim)
        #self.top_regression = nn.Linear(self.embed_dim, 4)
        self.down_regression_input = nn.Linear(self.embed_dim, self.embed_dim)
        self.down_regression = nn.Linear(self.embed_dim, 3)

        # Direct Regression Task
        #self.direct_input = nn.Linear(7, self.embed_dim)
        #self.direct_regression = nn.Linear(self.embed_dim, 1)
        
        # Track Classification
        #self.track_classification = nn.Linear(self.embed_dim, 3)

    def forward(self, lepton, MET, probe_jet, probe_jet_constituent, small_jet):
        
        # Feature initialization layers
        lepton_embedding = torch.unsqueeze(F.gelu(self.lepton_initializer(lepton)), dim=1)
        MET_embedding = torch.unsqueeze(F.gelu(self.MET_initializer(MET)), dim=1)
        probe_jet_embedding = torch.unsqueeze(F.gelu(self.probe_jet_initializer(probe_jet)), dim=1)
        probe_jet_constituent_embedding = F.gelu(self.probe_jet_constituent_initializer(probe_jet_constituent))
        small_jet_embedding = F.gelu(self.small_jet_initializer(small_jet))
        
        num_probe_jet = probe_jet_embedding.shape[1]
        num_constituents = probe_jet_constituent_embedding.shape[1]
        
        # Combine objects into single event tensor
        event_embedding = torch.cat([probe_jet_embedding, probe_jet_constituent_embedding, lepton_embedding, MET_embedding, small_jet_embedding], axis=1)
        
        # Transformer Stack
        probe_jet_embedding, probe_jet_constituent_embedding, event_embedding = self.stack1(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding, probe_jet_constituent_embedding, event_embedding = self.stack2(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)

        #probe_jet_embedding_top, probe_jet_constituent_embedding_top, event_embedding_top = self.stackTop(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        probe_jet_embedding_down, probe_jet_constituent_embedding_down, event_embedding_down = self.stackDown(probe_jet_embedding,probe_jet_constituent_embedding,event_embedding)
        
        # Track Classificiation
        #track_output = self.track_classification(probe_jet_constituent_embedding)
        
        #probe_jet_embedding_top  = torch.squeeze(probe_jet_embedding_top,1)
        probe_jet_embedding_down = torch.squeeze(probe_jet_embedding_down,1)
        
        # Get Top output
        #top_kinematics = F.gelu(self.top_regression_input(probe_jet_embedding_top))
        #top_kinematics = self.top_regression(top_kinematics)
        
        # Get Down output
        down_kinematics = F.gelu(self.down_regression_input(probe_jet_embedding_down))
        down_kinematics = F.tanh(self.down_regression(down_kinematics))
        
        # Direct Regression output
        #direct_embedding = torch.cat([top_kinematics, down_kinematics], axis=1)
        #direct_embedding = F.gelu(self.direct_input(direct_embedding))
        #costheta = self.direct_regression(direct_embedding)

        #return top_kinematics, down_kinematics, costheta, track_output
        return down_kinematics

class Model(nn.Module):  
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()   
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initiliazer
        self.lepton_initializer = nn.Linear(4, self.embed_dim)
        self.MET_initializer = nn.Linear(2, self.embed_dim)
        self.probe_jet_initializer = nn.Linear(4, self.embed_dim)
        self.probe_jet_constituent_initializer = nn.Linear(5, self.embed_dim)
        self.small_jet_initializer = nn.Linear(3, self.embed_dim)
           
        # Transformer Stack
        self.event_encoder1 = Encoder(self.embed_dim, self.num_heads)
        self.event_encoder2 = Encoder(self.embed_dim, self.num_heads)
        self.event_encoder3 = Encoder(self.embed_dim, self.num_heads)
        self.event_encoder4 = Encoder(self.embed_dim, self.num_heads)
        self.event_encoder5 = Encoder(self.embed_dim, self.num_heads)
        
        # Kinematics Regression
        self.top_regression_input = nn.Linear(self.embed_dim, self.embed_dim)
        self.top_regression = nn.Linear(self.embed_dim, 4)
        self.down_regression_input = nn.Linear(self.embed_dim, self.embed_dim)
        self.down_regression = nn.Linear(self.embed_dim, 3)

        # Direct Regression Task
        self.direct_input = nn.Linear(7, self.embed_dim)
        self.direct_regression = nn.Linear(self.embed_dim, 1)
        
        # Track Classification
        self.track_classification = nn.Linear(self.embed_dim, 3)

    def forward(self, lepton, MET, probe_jet, probe_jet_constituent, small_jet):
        
        # Feature initialization layers
        lepton_embedding = torch.unsqueeze(F.gelu(self.lepton_initializer(lepton)), dim=1)
        MET_embedding = torch.unsqueeze(F.gelu(self.MET_initializer(MET)), dim=1)
        probe_jet_embedding = torch.unsqueeze(F.gelu(self.probe_jet_initializer(probe_jet)), dim=1)
        probe_jet_constituent_embedding = F.gelu(self.probe_jet_constituent_initializer(probe_jet_constituent))
        small_jet_embedding = F.gelu(self.small_jet_initializer(small_jet))
        
        num_probe_jet = probe_jet_embedding.shape[1]
        num_constituents = probe_jet_constituent_embedding.shape[1]
        
        # Combine objects into single event tensor
        event_embedding = torch.cat([probe_jet_embedding, probe_jet_constituent_embedding, lepton_embedding, MET_embedding, small_jet_embedding], axis=1)
        
        # Event Level Attention
        event_embedding = self.event_encoder1(event_embedding,event_embedding,event_embedding)
        event_embedding = self.event_encoder2(event_embedding,event_embedding,event_embedding)
        event_embedding = self.event_encoder3(event_embedding,event_embedding,event_embedding)
        #event_embedding = self.event_encoder4(event_embedding,event_embedding,event_embedding)
        #event_embedding = self.event_encoder5(event_embedding,event_embedding,event_embedding)
        
        # Track Classificiation
        start_idx=num_probe_jet
        end_idx=start_idx+num_constituents
        track_tensor = event_embedding[:,start_idx:end_idx]
        track_output = self.track_classification(track_tensor)
        
        # Probe jet classification
        start_idx=0
        end_idx=num_probe_jet
        probe_jet_tensor = torch.squeeze(event_embedding[:,start_idx:end_idx], dim=1)
        
        # Get Top output
        top_kinematics = F.gelu(self.top_regression_input(probe_jet_tensor))
        top_kinematics = self.top_regression(top_kinematics)
        
        # Get Down output
        down_kinematics = F.gelu(self.down_regression_input(probe_jet_tensor))
        down_kinematics = F.tanh(self.down_regression(down_kinematics))
        down_kinematics = torch.cat([F.tanh(down_kinematics[:,0:3]), down_kinematics[:,3:]], axis=1)
        
        # Direct Regression output
        direct_embedding = torch.cat([top_kinematics, down_kinematics], axis=1)
        direct_embedding = F.gelu(self.direct_input(direct_embedding))
        costheta = self.direct_regression(direct_embedding)

        return top_kinematics, down_kinematics, costheta, track_output
