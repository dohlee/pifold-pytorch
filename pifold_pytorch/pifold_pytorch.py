import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl

from einops import rearrange
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


class PiGNNLayer(nn.Module):
    def __init__(self, d_emb):
        super().__init__()

        # (h_j || e_ji || h_i) to scalar
        self.att_mlp = nn.Linear(3 * d_emb, 1)
        # (e_ji || h_j) to d_emb
        self.node_mlp = nn.Linear(2 * d_emb, d_emb)
        # (h_j || e_ji || h_i) to e_emb
        self.edge_mlp = nn.Linear(3 * d_emb, d_emb)

        self.gate_mlp = nn.Linear(d_emb, d_emb)

    def forward(self, x):
        return x


class PiFold(pl.LightningModule):
    def __init__(self, d_node, d_edge, d_emb=128):
        super().__init__()

        self.d_emb = d_emb

        self.node_proj = nn.Linear(d_node, d_emb)
        self.edge_proj = nn.Linear(d_node, d_emb)


    def forward(self, x):
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
    pass
