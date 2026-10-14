from pathlib import Path


def load_class_names(dataset, setting, split):
    if dataset not in {"thumos", "anet"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if setting not in {50, 75} or split not in range(10):
        raise ValueError("Use setting 50 or 75 and split 0 through 9.")
    dataset_name = "THUMOS14" if dataset == "thumos" else "ActivityNet"
    split_dir = f"train_{setting}_test_{100 - setting}"
    path = (
        Path(__file__).resolve().parents[1]
        / "splits"
        / split_dir
        / dataset_name
        / "test"
        / f"split_{split}.list"
    )
    return path.read_text(encoding="utf-8").splitlines()
