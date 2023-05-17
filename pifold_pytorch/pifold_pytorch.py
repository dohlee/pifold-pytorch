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

        self.gate_mlp = nn.Linear(d_emb, d_emb)


    def forward(self, h, e, edge_index):
        # h: (# nodes, d_emb)
        # e: (# edges, d_emb)
        # edge_index: (2, # edges)
        hi, hj = h[edge_index[0]], h[edge_index[1]]

        hi_eij_hj = torch.cat( [hi, e, hj], dim=-1 )
        eij_hj = torch.cat( [e, hj], dim=-1 ) 

        #
        # Node update
        # 
        # Compute attention weights for each edge.
        w = self.att_mlp(hi_eij_hj) / torch.sqrt(self.d_head)
        att = w.softmax(dim=-1) # h, i, j

        # Compute node values.
        vj = self.node_mlp(eij_hj) # i, j, d_emb

        # Aggregate node values with attention weights
        # to update node features.
        h = einsum('bhij,bijd->bhid', att, vj)
        h = rearrange(h, 'b h i d -> (b i) (h d)')
        h = self.to_h(h) # Final linear projection.

        #
        # Edge update
        #
        e = self.edge_mlp(hi_eij_hj) 

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
    n_atoms = 64
    d_emb = 128
    n_neighbors = 30
    n_heads = 4

    att_mlp = nn.Sequential(
        nn.Linear(3 * d_emb, d_emb),
        nn.ReLU(),
        nn.Linear(d_emb, d_emb),
        nn.ReLU(),
        nn.Linear(d_emb, n_heads),
        Rearrange('(i j) h -> h i j', j=n_neighbors),
    )

    x = torch.randn(n_atoms * n_neighbors, d_emb)
    x = torch.cat([x, x, x], dim=-1)

    print(att_mlp(x).shape)
