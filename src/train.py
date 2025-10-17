from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Selected model: {cfg.models.results}")
    # load dataset
    data_loader = instantiate(cfg.data.loader)
    train_x, train_y = data_loader.load_train()

    # save count mapping for later use
    count_mapping_path = Path(cfg.data.encoder_path) / "count_mapping.pkl"
    data_loader.save_count_mapping(count_mapping_path)
    logger.info(f"Count mapping saved to {count_mapping_path}")

    # build model
    trainer = instantiate(cfg.models, logger=logger)
    # train model
    trainer.run_cv_training(train_x, train_y)

    # save model
    trainer.save_model(Path(cfg.models.model_path) / f"{cfg.models.results}.pkl")


if __name__ == "__main__":
    _main()
