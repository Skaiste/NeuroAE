from __future__ import annotations

import torch
from torch import nn


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        adj = adj.clamp(min=-1.0, max=1.0)
        adj = torch.abs(adj)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.matmul(adj / deg, x)
        out = self.self_linear(x) + self.neigh_linear(neigh)
        return torch.relu(self.norm(out))


class TopKPool(nn.Module):
    def __init__(self, in_dim, ratio):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.ratio = float(ratio)

    def forward(self, x, adj):
        scores = self.score(x).squeeze(-1)
        keep = max(2, int(x.shape[1] * self.ratio))
        topk_scores, topk_idx = torch.topk(scores, k=keep, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        pooled_x = torch.gather(x, 1, gather_idx) * torch.sigmoid(topk_scores).unsqueeze(-1)
        pooled_adj = torch.gather(adj, 1, topk_idx.unsqueeze(-1).expand(-1, -1, adj.shape[-1]))
        pooled_adj = torch.gather(pooled_adj, 2, topk_idx.unsqueeze(1).expand(-1, pooled_adj.shape[1], -1))
        reg_loss = (1.0 - torch.sigmoid(topk_scores).mean()) ** 2
        return pooled_x, pooled_adj, reg_loss


class BrainGNNClassifier(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        num_classes,
        hidden_dims=None,
        pool_ratios=None,
        dropout=0.2,
        aux_loss_weight=0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        pool_ratios = pool_ratios or [0.5, 0.5]
        if len(hidden_dims) < 2:
            raise ValueError("BrainGNNClassifier expects at least two hidden_dims.")
        self.conv1 = DenseGraphConv(node_feature_dim, hidden_dims[0])
        self.pool1 = TopKPool(hidden_dims[0], pool_ratios[0])
        self.conv2 = DenseGraphConv(hidden_dims[0], hidden_dims[1])
        self.pool2 = TopKPool(hidden_dims[1], pool_ratios[1])
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dims[1], num_classes),
        )
        self.aux_loss_weight = float(aux_loss_weight)

    def forward(self, node_features, adjacency):
        x = self.conv1(node_features, adjacency)
        x, adjacency, reg1 = self.pool1(x, adjacency)
        x = self.conv2(x, adjacency)
        x, adjacency, reg2 = self.pool2(x, adjacency)
        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        graph_repr = self.dropout(torch.cat([pooled_mean, pooled_max], dim=-1))
        logits = self.classifier(graph_repr)
        aux_loss = (reg1 + reg2) * self.aux_loss_weight
        return {"logits": logits, "aux_loss": aux_loss}

