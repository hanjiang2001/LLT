import os
import sys
from pathlib import Path

import hydra
import lightning as L
from lightning import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    dataset = hydra.utils.instantiate(cfg.data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    model = hydra.utils.instantiate(cfg.model)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,
        default_root_dir=cfg.paths.output_dir,
    )
    trainer.test(model, dataloaders=loader)


if __name__ == "__main__":
    main()
