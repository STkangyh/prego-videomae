import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import PACEDataset
from models import CNNLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PACEDataset(
    csv_path="data/METADATA_PACE_4_SAMPLES_ready.csv",
    video_dir="data/Archivo",
    num_frames=16
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = CNNLSTM(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(frames)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: {total_loss:.4f}")