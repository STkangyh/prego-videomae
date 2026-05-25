import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision import transforms
from PIL import Image


class PACEDataset(Dataset):
    def __init__(
        self,
        csv_path,
        video_dir,
        num_frames=16,
        image_size=224,
        partial_ratio=1.0
    ):
        self.df = pd.read_csv(csv_path, sep=";")
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.partial_ratio = partial_ratio

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.samples = []

        for _, row in self.df.iterrows():
            filename = row["File Name"]

            if pd.isna(filename):
                continue

            video_path = os.path.join(video_dir, filename)

            if not os.path.exists(video_path):
                continue

            error_type = row["Error Type"]

            # sample dataset currently contains errors
            label = 1

            self.samples.append({
                "video_path": video_path,
                "label": label,
                "error_type": error_type
            })

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))

        total_frames = len(vr)

        usable_frames = max(1, int(total_frames * self.partial_ratio))

        indices = np.linspace(
            0,
            usable_frames - 1,
            self.num_frames
        ).astype(int).tolist()

        frames = vr.get_batch(indices).asnumpy()

        processed_frames = []

        for frame in frames:
            img = Image.fromarray(frame)
            img = self.transform(img)
            processed_frames.append(img)

        return torch.stack(processed_frames)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frames = self._sample_frames(sample["video_path"])

        return {
            "frames": frames,
            "label": sample["label"],
            "error_type": sample["error_type"]
        }

dataset = PACEDataset(
    csv_path="data/METADATA_PACE_4_SAMPLES_ready.csv",
    video_dir="data/Archivo",
    num_frames=16,
    partial_ratio=1.0
)

sample = dataset[0]

print(sample["frames"].shape)
print(sample["label"])
print(sample["error_type"])