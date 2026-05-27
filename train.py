import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import NextStepDataset
from model import NextStepLSTM
from utils import collate_fn

import json


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for contexts, lengths, targets in loader:
            contexts = contexts.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(contexts)

            loss = criterion(logits, targets)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    acc = correct / total

    return total_loss / len(loader), acc


def main():
    dataset = NextStepDataset(
        "next_step_dataset.json",
        "action2idx.json"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    with open("action2idx.json") as f:
        vocab = json.load(f)

    model = NextStepLSTM(
        vocab_size=len(vocab)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for contexts, lengths, targets in train_loader:
            contexts = contexts.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits = model(contexts)

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()