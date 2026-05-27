import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    contexts = [
        item["context"]
        for item in batch
    ]

    targets = torch.stack([
        item["target"]
        for item in batch
    ])

    lengths = torch.tensor([
        len(x)
        for x in contexts
    ])

    padded = pad_sequence(
        contexts,
        batch_first=True,
        padding_value=0
    )

    return padded, lengths, targets