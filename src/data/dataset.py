import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.splits import load_class_names


class LLTDataset(Dataset):
    def __init__(self, dataset, feature_path, annotations_path, setting, split, metadata_path=None):
        self.feature_path = Path(feature_path)
        with open(annotations_path, encoding="utf-8") as stream:
            annotations = json.load(stream)
        class_names = set(load_class_names(dataset, setting, split))
        if dataset == "anet":
            video_names = pd.read_csv(metadata_path).iloc[:, 0].tolist()
        else:
            video_names = annotations.keys()
        self.video_names = [
            name
            for name in video_names
            if any(item["label"] in class_names for item in annotations[name]["annotations"])
        ]

    def __getitem__(self, index):
        video_name = self.video_names[index]
        features = np.load(self.feature_path / f"{video_name}.npy")
        return video_name, torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.video_names)
