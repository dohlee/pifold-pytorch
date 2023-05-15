import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from pifold_pytorch import PiFold, PiFoldDataset

def parse_argument():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_argument()

    model = PiFold() 

    train_set = PiFoldDataset()
    val_set = PiFoldDataset()
    test_set = PiFoldDataset()

    train_loader = DataLoader()
    val_loader = DataLoader()
    test_loader = DataLoader()

    trainer = pl.Trainer(
        accelerator='gpu',
        devies=1,
        max_epochs=100,
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
        test_loader,
    )


if __name__ == '__main__':
    main()