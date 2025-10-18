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
    data_loader = instantiate(
        cfg.models,
        logger=logger,
        num_features=cfg.features.num_features,
        cat_features=cfg.features.cat_features,
    )
    train_x, train_y = data_loader.load_train()

    # build model
    trainer = instantiate(
        cfg.models,
        logger=logger,
        features=cfg.features.num_features,
        cat_features=cfg.features.cat_features,
        n_splits=cfg.data.n_splits,
        split_type=cfg.data.split_type,
    )

    # train model
    trainer.run_cv_training(train_x, train_y)

    # save model
    trainer.save_model(Path(cfg.models.model_path))


if __name__ == "__main__":
    _main()
