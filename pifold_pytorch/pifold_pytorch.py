import math
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl

from einops import rearrange
from torch import einsum
from einops.layers.torch import Rearrange


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

    mask = (
        torch.zeros(l, bsz)
        .to(x.device)
        .scatter_(
            dim=1, index=batch_idx, src=torch.ones_like(batch_idx, dtype=torch.float32)
        )
    )
    bcount = mask.sum(dim=0)
    mask = mask.unsqueeze(-1).expand(-1, -1, d)  # i, bsz, d
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
            Rearrange("(i j) h -> h i j", j=n_neighbors),
        )
        # (e_ji || h_j) to d_emb
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, self.d_head),
            Rearrange("(i j) d -> i j d", j=n_neighbors),
        )
        self.to_h = nn.Linear(d_emb, d_emb, bias=False)

        # (h_j || e_ji || h_i) to e_emb
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.Dropout(0.1),
        )
        self.edge_bn = nn.BatchNorm1d(d_emb)

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
        assert len(h) == len(batch_idx)

        hi, hj = h[edge_idx[0]], h[edge_idx[1]]

        hi_eij_hj = torch.cat([hi, e, hj], dim=-1)
        eij_hj = torch.cat([e, hj], dim=-1)

        #
        # Node update
        #
        # Compute attention weights for each edge.
        w = self.att_mlp(hi_eij_hj) / math.sqrt(self.d_head)
        att = w.softmax(dim=-1)  # h, i, j

        # Compute node values.
        vj = self.node_mlp(eij_hj)  # i, j, d_emb

        # Aggregate node values with attention weights
        # to update node features.
        _h = einsum("hij,ijd->hid", att, vj)
        _h = rearrange(_h, "h i d -> i (h d)")
        _h = self.to_h(_h)  # Final linear projection.

        #
        # Edge update
        #
        e = self.edge_bn(e + self.edge_mlp(hi_eij_hj))

        #
        # Global context attention
        #
        c = scatter_mean(_h, batch_idx)

        gates = torch.sigmoid(self.gate_mlp(c))
        h = _h * gates[batch_idx]

        return h, e


class PiFold(pl.LightningModule):
    def __init__(
        self,
        d_node=165,
        d_edge=525,
        d_emb=128,
        d_rbf=16,
        n_heads=4,
        num_layers=10,
        n_virtual_atoms=3,
        n_neighbors=30,
        lr=1e-3,
        bsz=8,
    ):
        super().__init__()

        # Trainging parameters
        self.lr = lr
        self.num_layers = num_layers
        self.n_virtual_atoms = n_virtual_atoms
        self.bsz = bsz

        self.virtual_atoms = nn.Parameter(torch.randn(n_virtual_atoms, 3))
        self.d_rbf = d_rbf

        self.d_emb = d_emb
        self.node_proj = nn.Linear(d_node, d_emb)
        self.edge_proj = nn.Linear(d_edge, d_emb)

        self.layers = nn.ModuleList(
            [
                PiGNNLayer(d_emb, n_heads=n_heads, n_neighbors=n_neighbors)
                for _ in range(self.num_layers)
            ]
        )
        self.to_seq = nn.Linear(d_emb, 20)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, node, edge, edge_idx, batch_idx):
        h = self.node_proj(node)
        e = self.edge_proj(edge)

        for layer in self.layers:
            h, e = layer(h, e, edge_idx, batch_idx)

        logits = self.to_seq(h)
        return logits

    def training_step(self, batch):
        four_atom_coords, q = batch["four_atom_coords"], batch["q"]
        edge_idx, batch_idx = batch["edge_idx"], batch["batch_idx"]
        target = batch["aa_idx"]

        # Distance features depends on virtual atoms, so we need to compute them here
        node_dist_feat, edge_dist_feat = self.compute_dist_feat(
            four_atom_coords, q, edge_idx
        )
        # On the other hand, angle and direction features can be precomputed
        node_angle_feat = batch["node_angle_feat"]
        edge_angle_feat = batch["edge_angle_feat"]
        node_dir_feat = batch["node_dir_feat"]
        edge_dir_feat = batch["edge_dir_feat"]

        # Aggregate features
        node_feat = torch.cat([node_dist_feat, node_angle_feat, node_dir_feat], dim=-1)
        edge_feat = torch.cat([edge_dist_feat, edge_angle_feat, edge_dir_feat], dim=-1)

        out = self.forward(node_feat, edge_feat, edge_idx, batch_idx)

        loss = self.criterion(out, target)
        self.log_dict(
            {"train/loss": loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.bsz,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        four_atom_coords, q = batch["four_atom_coords"], batch["q"]
        edge_idx, batch_idx = batch["edge_idx"], batch["batch_idx"]
        target = batch["aa_idx"]

        # Distance features depends on virtual atoms, so we need to compute them here
        node_dist_feat, edge_dist_feat = self.compute_dist_feat(
            four_atom_coords, q, edge_idx
        )
        # On the other hand, angle and direction features can be precomputed
        node_angle_feat, edge_angle_feat = (
            batch["node_angle_feat"],
            batch["edge_angle_feat"],
        )
        node_dir_feat, edge_dir_feat = batch["node_dir_feat"], batch["edge_dir_feat"]

        # Aggregate features
        node_feat = torch.cat([node_dist_feat, node_angle_feat, node_dir_feat], dim=-1)
        edge_feat = torch.cat([edge_dist_feat, edge_angle_feat, edge_dir_feat], dim=-1)

        out = self.forward(node_feat, edge_feat, edge_idx, batch_idx)

        loss = self.criterion(out, target)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.bsz,
        )

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_dist_feat(self, four_atom_coords, q, edge_idx):
        """Given the four-atom coordinates (N, Ca, C, O) `four_atom_coords`,
        rotation matrices for corresponding residues `q`,
        and edge index towards top-k neighboring residues `edge_idx`
        return the node features and edge features.
        """
        CA_IDX = 0
        # assert not torch.isnan(self.virtual_atoms).any()
        virtual_atoms_norm = self.virtual_atoms / torch.norm(
            self.virtual_atoms, dim=1, keepdim=True
        ).unsqueeze(0).expand(four_atom_coords.shape[0], -1, -1)

        virtual_atom_coords = torch.bmm(q, virtual_atoms_norm.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # Rotate

        virtual_atom_coords = (
            virtual_atom_coords + four_atom_coords[:, None, CA_IDX]
        )  # Shift by Ca

        node_dist_feat = self.compute_node_dist_feat(
            four_atom_coords, virtual_atom_coords
        )
        edge_dist_feat = self.compute_edge_dist_feat(
            four_atom_coords, edge_idx, virtual_atom_coords
        )

        return node_dist_feat, edge_dist_feat

    def compute_node_dist_feat(self, fac, vac):
        # Compute distance between each pair of 'true' atoms.
        dist = torch.sqrt(((fac[:, None, :, :] - fac[:, :, None, :]) ** 2).sum(axis=-1))
        triu_indices = [1, 2, 3, 6, 7, 11]
        node_dist_feat = rbf(dist).view(-1, 4**2, self.d_rbf)
        node_dist_feat = node_dist_feat[:, triu_indices, :].view(
            -1, len(triu_indices) * self.d_rbf
        )

        # Compute distance between each pair of 'virtual' atoms.
        dist = torch.cdist(vac, vac)

        triu_indices = []
        idx = 0
        for i in range(self.n_virtual_atoms):
            for j in range(self.n_virtual_atoms):
                if i < j:
                    triu_indices.append(idx)
                idx += 1

        virtual_node_dist_feat = rbf(dist).view(
            -1, self.n_virtual_atoms**2, self.d_rbf
        )
        virtual_node_dist_feat = virtual_node_dist_feat[:, triu_indices, :].view(
            -1, len(triu_indices) * self.d_rbf
        )

        assert not node_dist_feat.isnan().any()

        return torch.cat(
            [node_dist_feat, virtual_node_dist_feat], dim=-1
        )  # (n_nodes, 6 + 3)
        # return node_dist_feat

    def compute_edge_dist_feat(self, four_atom_coords, edge_idx, virtual_atom_coords):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        # Compute distance between each pair of 'true' atoms in neighboring residues.
        four_atom_coords_i, four_atom_coords_j = (
            four_atom_coords[src_idx],
            four_atom_coords[dst_idx],
        )

        edge_dist_feat = torch.sqrt(
            (
                (four_atom_coords_i[:, None, :, :] - four_atom_coords_j[:, :, None, :])
                ** 2
            ).sum(axis=-1)
        )
        edge_dist_feat = rbf(edge_dist_feat)
        edge_dist_feat = edge_dist_feat.view(len(edge_dist_feat), -1)

        # Compute distance between each pair of 'virtual' atoms in neighboring residues.
        virtual_atom_coords_i, virtual_atom_coords_j = (
            virtual_atom_coords[src_idx],
            virtual_atom_coords[dst_idx],
        )

        virtual_edge_dist_feat = torch.sqrt(
            (
                (
                    virtual_atom_coords_i[:, None, :, :]
                    - virtual_atom_coords_j[:, :, None, :]
                )
                ** 2
            ).sum(axis=-1)
        )
        virtual_edge_dist_feat = rbf(virtual_edge_dist_feat)
        virtual_edge_dist_feat = edge_dist_feat.view(len(virtual_edge_dist_feat), -1)

        return torch.cat([edge_dist_feat, virtual_edge_dist_feat], axis=-1)


def rbf(dist, d_min=0, d_max=20, d_count=16):
    d_mu = torch.linspace(d_min, d_max, d_count).reshape(1, 1, 1, -1).to(dist.device)
    d_sigma = (d_max - d_min) / d_count
    dist = dist[:, :, :, None]

    return torch.exp(-((dist - d_mu) ** 2) / (2 * d_sigma**2))


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
    d_node = 16
    d_edge = 16

    node = torch.randn(1 * n_atoms, d_node)
    edge = torch.randn(1 * n_atoms * n_neighbors, d_edge)
    edge_idx = (
        torch.tensor(
            [
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 2],
                [2, 0],
                [2, 1],
            ]
        )
        .long()
        .T
    )

    batch_idx = torch.tensor([0, 0, 0]).long()

    model = PiFold(
        d_node=d_node,
        d_edge=d_edge,
        d_emb=d_emb,
        n_neighbors=n_neighbors,
        n_heads=n_heads,
    )
    out = model(node, edge, edge_idx, batch_idx)

    print(out.shape)

    # layer = PiGNNLayer(d_emb=d_emb, n_heads=n_heads, n_neighbors=n_neighbors)

    # h, e = layer.forward(h, e, edge_idx, batch_idx)
    # print(h.shape, e.shape)
