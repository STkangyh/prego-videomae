import json


def build_action_vocab(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    actions = set()

    for sample in dataset:
        for action in sample["context"]:
            actions.add(action)

        actions.add(sample["target_action"])

    actions = sorted(list(actions))

    action2idx = {
        action: idx for idx, action in enumerate(actions)
    }

    idx2action = {
        idx: action for action, idx in action2idx.items()
    }

    return action2idx, idx2action


if __name__ == "__main__":
    action2idx, idx2action = build_action_vocab(
        "next_step_dataset.json"
    )

    print(f"Vocabulary size: {len(action2idx)}")

    print(list(action2idx.items())[:20])

    with open("action2idx.json", "w") as f:
        json.dump(action2idx, f, indent=2)

    with open("idx2action.json", "w") as f:
        json.dump(idx2action, f, indent=2)