import pandas as pd
import json
from collections import Counter

# ── Config ────────────────────────────────────────────────
CSV_PATH         = "EPIC_100_train.csv"
MAX_GAP          = 450   # 시간 세그멘테이션 최대 프레임 간격
WINDOW_SIZE      = 5     # 슬라이딩 윈도우 크기
MIN_UNIQUE_VERBS = 2     # 절차적 시퀀스 판별 최소 고유 동사 수
# ──────────────────────────────────────────────────────────

ALLOWED_VERBS = {
    "take",
    "put",
    "put-on",
    "put-in",
    "put-into",
    "pour",
    "stir",
    "mix",
    "wash",
    "cut",
    "slice",
    "grate",
    "peel",
    "insert",
    "remove",
    "throw-into",
    "drop"
}


def deduplicate(sequence):
    result = []

    for item in sequence:
        if not result or result[-1]["action"] != item["action"]:
            result.append(item)

    return result


def sliding_windows(sequence):
    windows = []

    if len(sequence) < WINDOW_SIZE:
        return windows

    for i in range(len(sequence) - WINDOW_SIZE + 1):
        windows.append(sequence[i:i + WINDOW_SIZE])

    return windows


def is_procedural(chunk, min_unique_verbs=2, max_dominant_ratio=0.6):
    verbs = [item["verb"] for item in chunk]

    unique_verbs = len(set(verbs))

    counts = Counter(verbs)
    dominant_ratio = max(counts.values()) / len(verbs)

    return (
        unique_verbs >= min_unique_verbs and
        dominant_ratio <= max_dominant_ratio
    )


def segment_by_time(group):
    segments = []

    current = []
    prev_stop = None

    for _, row in group.iterrows():
        item = {
            "action": row["verb"],
            "verb": row["verb"],
            "noun": row["noun"],
            "start_frame": row["start_frame"],
            "stop_frame": row["stop_frame"],
            "video_id": row["video_id"]
        }

        if prev_stop is None:
            current.append(item)

        else:
            gap = row["start_frame"] - prev_stop

            if gap <= MAX_GAP:
                current.append(item)
            else:
                if len(current) > 0:
                    segments.append(current)
                current = [item]

        prev_stop = row["stop_frame"]

    if current:
        segments.append(current)

    return segments


def build_workflows(csv_path):
    df = pd.read_csv(csv_path)

    df = df[df["verb"].isin(ALLOWED_VERBS)]

    workflows = []

    for video_id, group in df.groupby("video_id"):
        segments = segment_by_time(group)

        for segment in segments:
            segment = deduplicate(segment)

            chunks = sliding_windows(segment)

            for chunk in chunks:
                if is_procedural(chunk):
                    workflows.append(chunk)

    return workflows


def build_next_step_dataset(workflows):
    dataset = []

    for workflow in workflows:
        for i in range(1, len(workflow)):
            sample = {
                "video_id": workflow[i]["video_id"],
                "context": [x["action"] for x in workflow[:i]],
                "target_action": workflow[i]["action"],
                "target_start_frame": workflow[i]["start_frame"],
                "target_stop_frame": workflow[i]["stop_frame"]
            }

            dataset.append(sample)

    return dataset


if __name__ == "__main__":
    workflows = build_workflows(CSV_PATH)

    print(f"Workflows: {len(workflows)}")

    dataset = build_next_step_dataset(workflows)

    print(f"Training samples: {len(dataset)}")

    print(dataset[:5])

    with open("next_step_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

