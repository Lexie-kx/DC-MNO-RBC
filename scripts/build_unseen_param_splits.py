# scripts/build_unseen_param_splits.py

import json
import re
from pathlib import Path


def parse_group(group_name: str):
    """
    Parse group name like:
        ra_1e6_pr_0.5
        ra_1e7_pr_1
        ra_1e8_pr_2

    Return:
        ra_str, pr_str
    """
    pattern = r"ra_(.+)_pr_(.+)"
    match = re.match(pattern, group_name)
    if match is None:
        raise ValueError(f"Invalid group name format: {group_name}")

    ra_str = match.group(1)
    pr_str = match.group(2)
    return ra_str, pr_str


def collect_all_groups(iid_split):
    """
    Collect all unique groups from iid split.
    """
    groups = []
    seen = set()

    for split_name in ["train", "val", "test"]:
        for item in iid_split[split_name]:
            group = item["group"]
            if group not in seen:
                groups.append(group)
                seen.add(group)

    return groups


def make_entry(group, trajectories):
    return {
        "group": group,
        "trajectories": trajectories
    }


def build_unseen_ra_split(all_groups):
    """
    unseen Ra setting:
        train Ra: 1e6, 1e7
        val Ra:   1e6, 1e7
        test Ra:  1e8

    train trajectories: 0-7
    val trajectories:   8
    test trajectories:  0-9
    """
    train = []
    val = []
    test = []

    for group in all_groups:
        ra_str, pr_str = parse_group(group)

        if ra_str in ["1e6", "1e7"]:
            train.append(make_entry(group, list(range(0, 8))))
            val.append(make_entry(group, [8]))

        elif ra_str == "1e8":
            test.append(make_entry(group, list(range(0, 10))))

        else:
            raise ValueError(f"Unexpected Ra value: {ra_str} in group {group}")

    return {
        "split_type": "unseen_ra",
        "description": (
            "Cross-parameter split. Train/val use Ra=1e6 and Ra=1e7; "
            "test uses unseen Ra=1e8. Train trajectories 0-7, val trajectory 8, "
            "test trajectories 0-9 for unseen Ra groups."
        ),
        "train": train,
        "val": val,
        "test": test
    }


def build_unseen_pr_split(all_groups):
    """
    unseen Pr setting:
        train Pr: 0.5, 1
        val Pr:   0.5, 1
        test Pr:  2

    train trajectories: 0-7
    val trajectories:   8
    test trajectories:  0-9
    """
    train = []
    val = []
    test = []

    for group in all_groups:
        ra_str, pr_str = parse_group(group)

        if pr_str in ["0.5", "1"]:
            train.append(make_entry(group, list(range(0, 8))))
            val.append(make_entry(group, [8]))

        elif pr_str == "2":
            test.append(make_entry(group, list(range(0, 10))))

        else:
            raise ValueError(f"Unexpected Pr value: {pr_str} in group {group}")

    return {
        "split_type": "unseen_pr",
        "description": (
            "Cross-parameter split. Train/val use Pr=0.5 and Pr=1; "
            "test uses unseen Pr=2. Train trajectories 0-7, val trajectory 8, "
            "test trajectories 0-9 for unseen Pr groups."
        ),
        "train": train,
        "val": val,
        "test": test
    }


def summarize_split(split):
    print(f"\n===== {split['split_type']} =====")
    for split_name in ["train", "val", "test"]:
        groups = split[split_name]
        n_groups = len(groups)
        n_traj = sum(len(item["trajectories"]) for item in groups)

        print(f"{split_name}: {n_groups} groups, {n_traj} group-trajectories")

        for item in groups:
            print(f"  {item['group']}: {item['trajectories']}")


def main():
    project_root = Path(__file__).resolve().parents[1]

    iid_path = project_root / "data" / "splits" / "iid_split.json"
    output_dir = project_root / "data" / "splits"

    unseen_ra_path = output_dir / "unseen_ra_split.json"
    unseen_pr_path = output_dir / "unseen_pr_split.json"

    if not iid_path.exists():
        raise FileNotFoundError(f"Cannot find iid split file: {iid_path}")

    with open(iid_path, "r", encoding="utf-8") as f:
        iid_split = json.load(f)

    all_groups = collect_all_groups(iid_split)

    print("Detected groups:")
    for group in all_groups:
        ra_str, pr_str = parse_group(group)
        print(f"  {group} -> Ra={ra_str}, Pr={pr_str}")

    unseen_ra_split = build_unseen_ra_split(all_groups)
    unseen_pr_split = build_unseen_pr_split(all_groups)

    summarize_split(unseen_ra_split)
    summarize_split(unseen_pr_split)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(unseen_ra_path, "w", encoding="utf-8") as f:
        json.dump(unseen_ra_split, f, indent=4)

    with open(unseen_pr_path, "w", encoding="utf-8") as f:
        json.dump(unseen_pr_split, f, indent=4)

    print("\nSaved:")
    print(f"  {unseen_ra_path}")
    print(f"  {unseen_pr_path}")


if __name__ == "__main__":
    main()