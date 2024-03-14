from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig

from data import load_train_dataset
from models import bulid_model
from utils import reduce_mem_usage


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # load dataset
        train_x, train_y = load_train_dataset(cfg)
        train_x = reduce_mem_usage(train_x)

        # choose trainer
        trainer = bulid_model(cfg)

        # train model
        trainer.run_cv_training(train_x, train_y)

        # save model
        trainer.save_model(Path(cfg.models.path) / f"{cfg.models.results}.pkl")


if __name__ == "__main__":
    _main()
