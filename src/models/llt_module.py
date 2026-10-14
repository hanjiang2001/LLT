import json
from pathlib import Path

import torch
from lightning import LightningModule

from src.evaluate import evaluate


class LLTModule(LightningModule):
    def __init__(self, net, dataset, split, setting, fps_path, annotations_path, output_dir):
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.split = split
        self.setting = setting
        self.fps_path = fps_path
        self.annotations_path = annotations_path
        self.output_dir = Path(output_dir)
        self.predictions = {}

    @torch.inference_mode(False)
    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        video_name, predictions = self.net(batch)
        self.predictions[video_name] = predictions

    def on_test_epoch_end(self):
        metrics = evaluate(
            self.dataset,
            self.predictions,
            self.net.class_names,
            self.annotations_path,
            self.fps_path,
            self.output_dir,
        )
        self.log("avg_AP", metrics["avg_AP"])
        metrics.update(dataset=self.dataset, split=self.split, setting=self.setting)
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
