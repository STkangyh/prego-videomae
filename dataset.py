import os
import pandas as pd
from torch.utils.data import Dataset

class PACEDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.df = pd.read_csv(csv_path, sep=";")
        self.video_dir = video_dir

        self.samples = []

        for _, row in self.df.iterrows():
            filename = row["File Name"]
            error_type = row["Error Type"]

            if pd.isna(filename):
                continue

            video_path = os.path.join(video_dir, filename)

            label = 1  # current sample only contains errors

            self.samples.append({
                "video_path": video_path,
                "label": label,
                "error_type": error_type
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = PACEDataset(
    csv_path="data/METADATA_PACE_4_SAMPLES_ready.csv",
    video_dir="Archivo"
)

print(len(dataset))
print(dataset[0])