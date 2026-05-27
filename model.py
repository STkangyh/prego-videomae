import torch
import torch.nn as nn


class NextStepLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):
        x = self.embedding(x)

        out, (h, c) = self.lstm(x)

        logits = self.classifier(h[-1])

        return logits