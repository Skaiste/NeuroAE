import torch
import torch.nn as nn


class TemporalAttentionPooling(nn.Module):
    """ Model focuses on time points that are more important """
    def __init__(self, latent_dim):
        super().__init__()
        self.attn = nn.Linear(latent_dim, 1)

    def forward(self, z): # z: (B, L, T)
        z_t = z.transpose(1, 2)      # (B, T, L)
        scores = self.attn(z_t)      # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * z_t).sum(dim=1)  # (B, L)
        return pooled


class GatedTemporalAttentionPooling(nn.Module):
    """ Richer version of attention pooling, uses two projections:
         - one to model the content
         - one to act like a gate (suppress or amplify parts of the timepoint representation)
        It can capture:
         - this timepoint is important only if certain latent features are active
         - interactions between latent content and importance
         - more selective weighting over time
        it often gives sharper and more meaningful attention weights.
    """
    def __init__(self, latent_dim: int, attn_dim: int):
        super().__init__()
        print(f"GatedTemporalAttentionPooling {latent_dim=} {attn_dim=}")
        self.V = nn.Linear(latent_dim, attn_dim)
        self.U = nn.Linear(latent_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, z):   # z: (B, L, T)
        z_t = z.transpose(1, 2)  # (B, T, L)
        v = torch.tanh(self.V(z_t))        # (B, T, A)
        u = torch.sigmoid(self.U(z_t))     # (B, T, A)
        h = v * u                          # (B, T, A)
        scores = self.w(h)                 # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # (B, T, 1)
        pooled = (weights * z_t).sum(dim=1)      # (B, L)
        return pooled, weights.squeeze(-1)
    

"""
    Prediction head models
"""
class PredHeadAvg(nn.Module):
    """ Most simple model
         - averaging across time point dimension 
         - applying linear layer to predict levels
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(latent_dim, output_dim, bias=True)

    def forward(self, z):
        x = z.mean(dim=-1)
        return self.head(x)
    
class PredHeadConv(nn.Module):
    """ Convolutional model
        - temporal conv model with average pooling
        - linear layer for level prediction
    """
    def __init__(self, latent_dim, output_dim, with_hidden=True):
        super().__init__()
        # since selected latent dimension is usually a small number
        # we can double it, another suggestion would be to get the middle number
        # in between latent and output dimensions, but that would bloat the model
        last_dim = latent_dim
        if with_hidden:
            self.hidden_dim = latent_dim * 2
            self.temporal = nn.Sequential(
                nn.Conv1d(latent_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            last_dim = self.hidden_dim
        else:
            self.temporal = nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_dim, output_dim)
        )
    def forward(self, z):
        x = self.temporal(z)
        return self.head(x)
    

class PredHeadTemporalPool(nn.Module):
    """ Temporal pooling model
         - use temporal pooling to reduce time point dimension to 1
         - applying linear layer to predict levels
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.pool = TemporalAttentionPooling(latent_dim)
        self.head = nn.Linear(latent_dim, output_dim, bias=True)

    def forward(self, z):
        x = self.pool(z)
        return self.head(x)
    
    
class PredHeadGatedTemporalPool(nn.Module):
    """ Gated Temporal pooling model
         - use gated temporal pooling to reduce time point dimension to 1
         - applying linear layer to predict levels
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # like the convolutional hidden dim, same applies
        self.attention_dim = latent_dim * 2
        self.pool = GatedTemporalAttentionPooling(latent_dim, self.attention_dim)
        self.head = nn.Linear(latent_dim, output_dim, bias=True)

    def forward(self, z):
        x, _ = self.pool(z)
        return self.head(x)
