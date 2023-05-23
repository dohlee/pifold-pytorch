import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import os

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor

from pifold_pytorch import PiFold, PiFoldDataset


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", required=True, help="Metadata file for training data"
    )
    parser.add_argument(
        "--val", required=True, help="Metadata file for validation data"
    )
    parser.add_argument("--test", required=True, help="Metadata file for test data")
    parser.add_argument(
        "-d", "--data-dir", required=True, help="Directory containing preprocessed data"
    )
    parser.add_argument("-b", "--bsz", type=int, default=8, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "-n", "--num-layers", type=int, default=10, help="Number of PiGNN layers"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Don't use wandb for logging",
    )
    return parser.parse_args()


def collate(data):
    batch = {}

    four_atom_coords, q = [], []
    edge_idx, aa_idx, batch_idx = [], [], []
    node_angle_feat, edge_angle_feat = [], []
    node_dir_feat, edge_dir_feat = [], []

    for i, d in enumerate(data):
        assert len(d["aa_idx"]) == len(d["node_angle_feat"])
        assert len(d["aa_idx"]) == len(d["node_dir_feat"])

        num_residues = len(d["aa_idx"])

        four_atom_coords.append(d["four_atom_coords"])
        q.append(d["q"])
        edge_idx.append(d["edge_idx"])
        aa_idx.append(d["aa_idx"])

        batch_idx.append(torch.ones(num_residues).long() * i)

        node_angle_feat.append(d["node_angle_feat"])
        edge_angle_feat.append(d["edge_angle_feat"])
        node_dir_feat.append(d["node_dir_feat"])
        edge_dir_feat.append(d["edge_dir_feat"])

    batch["four_atom_coords"] = torch.cat(four_atom_coords)
    batch["q"] = torch.cat(q)
    batch["edge_idx"] = torch.cat(edge_idx, axis=1)
    batch["aa_idx"] = torch.cat(aa_idx)
    batch["batch_idx"] = torch.cat(batch_idx)

    batch["node_angle_feat"] = torch.cat(node_angle_feat).nan_to_num(0.0)
    batch["edge_angle_feat"] = torch.cat(edge_angle_feat).nan_to_num(0.0)
    batch["node_dir_feat"] = torch.cat(node_dir_feat)
    batch["edge_dir_feat"] = torch.cat(edge_dir_feat)

    for k in [
        "four_atom_coords",
        "q",
        "edge_idx",
        "aa_idx",
        "batch_idx",
        "node_angle_feat",
        "edge_angle_feat",
        "node_dir_feat",
        "edge_dir_feat",
    ]:
        if torch.isnan(batch[k]).any():
            for _ in range(10):
                print(k)

    return batch


def main():
    torch.set_float32_matmul_precision("high")

    args = parse_argument()
    pl.seed_everything(args.seed)

    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            project="pifold-pytorch",
            entity="dohlee",
        )

    model = PiFold(
        num_layers=args.num_layers,
        lr=args.learning_rate,
    )

    train_meta = pd.read_csv(args.train)
    val_meta = pd.read_csv(args.val)
    test_meta = pd.read_csv(args.test)

    train_set = PiFoldDataset(meta=train_meta, data_dir=args.data_dir)
    val_set = PiFoldDataset(meta=val_meta, data_dir=args.data_dir)
    test_set = PiFoldDataset(meta=test_meta, data_dir=args.data_dir)

    train_loader = DataLoader(
        train_set,
        batch_size=args.bsz,
        collate_fn=collate,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.bsz,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bsz,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        logger=logger,
    )

    # NOTE: ignore val_dataloader for now in reproduction
    trainer.fit(
        model,
        train_loader,
        test_loader,
    )


if __name__ == "__main__":
    main()
