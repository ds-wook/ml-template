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
        cfg.data,
        logger=logger,
        num_features=cfg.features.num_features,
        cat_features=cfg.features.cat_features,
    )
    train_x, train_y = data_loader.load_train_dataset()

    # merge num and cat features
    features = [*cfg.features.num_features, *cfg.features.cat_features]

    # build model
    trainer = instantiate(
        cfg.models,
        logger=logger,
        features=features,
        cat_features=cfg.features.cat_features,
        n_splits=cfg.models.n_splits,
    )

    # train model
    trainer.run_cv_training(train_x, train_y)

    # save model
    trainer.save_model(Path(cfg.models.model_path))


if __name__ == "__main__":
    _main()
