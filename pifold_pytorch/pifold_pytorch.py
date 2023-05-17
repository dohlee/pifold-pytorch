import math
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl

from einops import rearrange
from torch import einsum
from einops.layers.torch import Rearrange


class NodeMLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EdgeMLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GateMLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)
    
def scatter_mean(x, batch_idx):
    l, bsz = len(batch_idx), batch_idx.max().item() + 1
    d = x.size(-1)

    batch_idx = batch_idx.unsqueeze(0).T

    mask = torch.zeros(l, bsz).scatter_(
        dim=1,
        index=batch_idx,
        src=torch.ones_like(batch_idx, dtype=torch.float32)
    )
    bcount = mask.sum(dim=0)
    mask = mask.unsqueeze(-1).expand(-1, -1, d) # i, bsz, d

    # masking
    x = x.unsqueeze(1).expand(-1, bsz, -1) * mask

    # average
    x = x.sum(dim=0) / bcount.unsqueeze(1)
    return x

class PiGNNLayer(nn.Module):
    def __init__(self, d_emb, n_heads, n_neighbors):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_emb // n_heads
        self.n_neighbors = n_neighbors

        # (h_j || e_ji || h_i) to scalar
        self.att_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, n_heads),
            Rearrange('(i j) h -> h i j', j=n_neighbors),
        )
        # (e_ji || h_j) to d_emb
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            Rearrange('(i j) d -> i j d', j=n_neighbors)
        )
        self.to_h = nn.Linear(d_emb, d_emb, bias=False)

        # (h_j || e_ji || h_i) to e_emb
        self.edge_mlp = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Linear(3 * d_emb, d_emb),
                    nn.GELU(),
                    nn.Linear(d_emb, d_emb),
                    nn.GELU(),
                    nn.Linear(d_emb, d_emb),
                )
            ),
            nn.BatchNorm1d(d_emb),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
        )


    def forward(self, h, e, edge_idx, batch_idx):
        # h: (# nodes, d_emb)
        # e: (# edges, d_emb)
        # edge_index: (2, # edges)
        hi, hj = h[edge_idx[0]], h[edge_idx[1]]

        hi_eij_hj = torch.cat( [hi, e, hj], dim=-1 )
        eij_hj = torch.cat( [e, hj], dim=-1 ) 

        #
        # Node update
        # 
        # Compute attention weights for each edge.
        w = self.att_mlp(hi_eij_hj) / math.sqrt(self.d_head)
        att = w.softmax(dim=-1) # h, i, j

        # Compute node values.
        vj = self.node_mlp(eij_hj) # i, j, d_emb

        # Aggregate node values with attention weights
        # to update node features.
        _h = einsum('hij,ijd->hid', att, vj)
        print(_h.shape)
        _h = rearrange(_h, 'h i d -> i (h d)')
        _h = self.to_h(_h) # Final linear projection.

        #
        # Edge update
        #
        e = self.edge_mlp(hi_eij_hj) 

        #
        # Global context attention
        #
        c = scatter_mean(_h, batch_idx)
        h = _h * torch.sigmoid(self.gate_mlp(c))

        return h


class PiFold(pl.LightningModule):
    def __init__(self, d_node, d_edge, d_emb=128):
        super().__init__()

        self.d_emb = d_emb

        self.node_proj = nn.Linear(d_node, d_emb)
        self.edge_proj = nn.Linear(d_edge, d_emb)

        self.layers = nn.ModuleList([
            PiGNNLayer(d_emb, n_heads=4, n_neighbors=30)
            for _ in range(10)
        ])

    def forward(self, node, edge, edge_index):
        h = self.node_proj(node)
        e = self.edge_proj(edge)

        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # n_atoms = 64
    # d_emb = 128
    # n_neighbors = 30
    # n_heads = 4

    # att_mlp = nn.Sequential(
    #     nn.Linear(3 * d_emb, d_emb),
    #     nn.ReLU(),
    #     nn.Linear(d_emb, d_emb),
    #     nn.ReLU(),
    #     nn.Linear(d_emb, n_heads),
    #     Rearrange('(i j) h -> h i j', j=n_neighbors),
    # )

    # x = torch.randn(n_atoms * n_neighbors, d_emb)
    # x = torch.cat([x, x, x], dim=-1)

    # print(att_mlp(x).shape)

    # x = torch.tensor([
    #     [1, 1, 1],
    #     [2, 2, 2],
    #     [1, 1, 1],
    #     [2, 2, 2],
    #     [1, 1, 1],
    #     [2, 2, 2],
    #     [3, 3, 3],
    # ]).float()

    # batch_idx = torch.tensor([0, 1, 0, 1, 0, 2, 2]).long()

    # s = scatter_mean(x, batch_idx)
    # print(s)

    n_atoms = 3
    d_emb = 128
    n_neighbors = 2
    n_heads = 1

    h = torch.randn(1 * n_atoms, 128)
    e = torch.randn(1 * n_atoms * n_neighbors, 128)
    edge_idx = torch.tensor([
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 2],
        [2, 0],
        [2, 1],
    ]).long().T

    batch_idx = torch.tensor([0, 0, 0, 0]).long()

    layer = PiGNNLayer(d_emb=d_emb, n_heads=n_heads, n_neighbors=n_neighbors)
    layer.forward(h, e, edge_idx, batch_idx)