from collections import Counter
import json

with open("next_step_dataset.json") as f:
    data = json.load(f)

counter = Counter()

for sample in data:
    participant = sample["video_id"].split("_")[0]
    counter[participant] += 1

print(counter)