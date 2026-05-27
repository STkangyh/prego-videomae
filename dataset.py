import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NextStepDataset(Dataset):
    def __init__(self, dataset_path, vocab_path):
        with open(dataset_path) as f:
            self.samples = json.load(f)

        with open(vocab_path) as f:
            self.action2idx = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        context_ids = [
            self.action2idx[action]
            for action in sample["context"]
        ]

        target_id = self.action2idx[
            sample["target_action"]
        ]

        return {
            "context": torch.tensor(
                context_ids,
                dtype=torch.long
            ),
            "target": torch.tensor(
                target_id,
                dtype=torch.long
            )
        }
