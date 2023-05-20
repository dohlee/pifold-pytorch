import torch
from pathlib import Path
from torch.utils.data import Dataset

class PiFoldDataset(Dataset):
    def __init__(self, meta, data_dir):
        self.records = meta.to_records()
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        pt_fp = self.data_dir / record.pt_path
        return torch.load(pt_fp)