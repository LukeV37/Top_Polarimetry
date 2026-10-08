import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_ANALYSIS_TYPES = ("top", "down", "bottom")
VALID_LOSS_TYPES = ("mse", "cossim", "vmf")
LOSS_TYPE_ALIASES = {
    "cos": "cossim",
    "cosine": "cossim",
    "cos_sim": "cossim",
    "cosinesim": "cossim",
    "cosine_similarity": "cossim",
    "cosine_embedding": "cossim",
}


def validate_analysis_type(analysis_type):
    analysis_type = str(analysis_type).lower()
    if analysis_type not in VALID_ANALYSIS_TYPES:
        raise ValueError(
            f"analysis_type must be one of {VALID_ANALYSIS_TYPES}, got {analysis_type!r}"
        )
    return analysis_type


def validate_loss_type(loss_type, analysis_type):
    analysis_type = validate_analysis_type(analysis_type)
    if loss_type is None or str(loss_type).strip() == "":
        return "mse" if analysis_type == "top" else "cossim"

    loss_type = str(loss_type).lower().replace("-", "_")
    loss_type = LOSS_TYPE_ALIASES.get(loss_type, loss_type)

    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(
            f"loss_type must be one of {VALID_LOSS_TYPES}, got {loss_type!r}"
        )
    if analysis_type == "top" and loss_type != "mse":
        raise ValueError("analysis_type='top' always uses loss_type='mse'")

    return loss_type


class VMFHead(nn.Module):
    """Predict vMF mean direction and concentration for 3-vector tasks."""

    def __init__(self, in_dim):
        super(VMFHead, self).__init__()
        self.fc = nn.Linear(in_dim, 4)

    def forward(self, h):
        out = self.fc(h)
        mu = F.normalize(out[:, :3], dim=-1)
        kappa = F.softplus(out[:, 3]) + 1e-4
        return mu, kappa


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Encoder, self).__init__()
        self.pre_norm_Q = nn.LayerNorm(embed_dim)
        self.pre_norm_K = nn.LayerNorm(embed_dim)
        self.pre_norm_V = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True, dropout=0.25
        )
        self.post_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

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
        jet_trk_embedding = self.jet_trk_encoder(
            jet_trk_embedding, jet_trk_embedding, jet_trk_embedding
        )
        # Cross Attention (Local)
        jet_embedding = self.jet_trk_cross_encoder(
            jet_embedding, jet_trk_embedding, jet_trk_embedding
        )
        # Track Attention
        trk_embedding = self.trk_encoder(trk_embedding, trk_embedding, trk_embedding)
        # Cross Attention (Global)
        jet_embedding = self.trk_cross_encoder(jet_embedding, trk_embedding, trk_embedding)
        return jet_embedding, jet_trk_embedding, trk_embedding


class Model(nn.Module):
    """Single-task model selected by analysis_type.

    analysis_type="top" predicts a 4-vector and always uses MSE.
    analysis_type="down" or "bottom" predicts a normalized 3-vector direction.
    For down/bottom, loss_type can be "mse", "cossim", or "vmf".

    The previous multi-output heads (track classification, direct/costheta
    regression, simultaneous top/quark heads) are intentionally not constructed
    here so each checkpoint is an independent single-task model.
    """

    def __init__(self, embed_dim, num_heads, analysis_type="down", loss_type=None):
        super(Model, self).__init__()

        self.analysis_type = validate_analysis_type(analysis_type)
        self.loss_type = validate_loss_type(loss_type, self.analysis_type)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = 4 if self.analysis_type == "top" else 3

        # Initializers
        self.probe_jet_initializer = nn.Linear(4, self.embed_dim)
        self.probe_jet_constituent_initializer = nn.Linear(4, self.embed_dim)
        self.event_initializer = nn.Linear(4, self.embed_dim)

        # Shared feature extraction for the selected task.
        self.stack1 = Stack(self.embed_dim, self.num_heads)
        self.stack2 = Stack(self.embed_dim, self.num_heads)
        self.task_stack = Stack(self.embed_dim, self.num_heads)

        # Single task-specific regression head. vMF uses an extra concentration
        # output, but is still a single-task down/bottom model.
        if self.loss_type == "vmf":
            self.task_regression = VMFHead(self.embed_dim)
        else:
            self.task_regression = nn.Linear(self.embed_dim, self.output_dim)

    def _normalize_if_direction_task(self, output):
        if isinstance(output, tuple):
            return output
        analysis_type = getattr(self, "analysis_type", "down")
        if analysis_type in ("down", "bottom"):
            output = F.normalize(output, dim=1)
        return output

    def _legacy_forward(self, probe_jet_embedding, probe_jet_constituent_embedding, event_embedding):
        """Compatibility for older serialized multi-output checkpoints.

        New models do not instantiate these legacy layers. This branch lets an
        older checkpoint still be evaluated after evaluate.py sets analysis_type.
        """
        analysis_type = getattr(self, "analysis_type", "down")

        if analysis_type == "top":
            probe_jet_embedding_NEW, _, _ = self.stackTop(
                probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
            )
            probe_jet_embedding = torch.squeeze(probe_jet_embedding + probe_jet_embedding_NEW, 1)
            return self.top_regression(probe_jet_embedding)

        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stackQuark1(
            probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
        )
        probe_jet_embedding = probe_jet_embedding + probe_jet_embedding_NEW
        probe_jet_constituent_embedding = probe_jet_constituent_embedding + probe_jet_constituent_embedding_NEW
        event_embedding = event_embedding + event_embedding_NEW

        probe_jet_embedding_NEW, _, _ = self.stackQuark2(
            probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
        )
        probe_jet_embedding = torch.squeeze(probe_jet_embedding + probe_jet_embedding_NEW, 1)
        output = self.quark_regression(probe_jet_embedding)
        return F.normalize(output, dim=1)

    def forward(self, probe_jet, probe_jet_constituent, event_tensor):
        # Feature initialization layers
        probe_jet_embedding = F.gelu(self.probe_jet_initializer(probe_jet))
        probe_jet_constituent_embedding = F.gelu(
            self.probe_jet_constituent_initializer(probe_jet_constituent)
        )
        event_embedding = F.gelu(self.event_initializer(event_tensor))

        # Transformer encoder stack
        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stack1(
            probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
        )
        probe_jet_embedding = probe_jet_embedding + probe_jet_embedding_NEW
        probe_jet_constituent_embedding = probe_jet_constituent_embedding + probe_jet_constituent_embedding_NEW
        event_embedding = event_embedding + event_embedding_NEW

        probe_jet_embedding_NEW, probe_jet_constituent_embedding_NEW, event_embedding_NEW = self.stack2(
            probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
        )
        probe_jet_embedding = probe_jet_embedding + probe_jet_embedding_NEW
        probe_jet_constituent_embedding = probe_jet_constituent_embedding + probe_jet_constituent_embedding_NEW
        event_embedding = event_embedding + event_embedding_NEW

        # New single-task model path.
        if hasattr(self, "task_stack"):
            probe_jet_embedding_NEW, _, _ = self.task_stack(
                probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
            )
            probe_jet_embedding = torch.squeeze(probe_jet_embedding + probe_jet_embedding_NEW, 1)
            output = self.task_regression(probe_jet_embedding)
            return self._normalize_if_direction_task(output)

        return self._legacy_forward(
            probe_jet_embedding, probe_jet_constituent_embedding, event_embedding
        )
