from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Selected model: {cfg.models.results}")

    # load dataset
    data_loader = hydra.utils.instantiate(
        cfg.data,
        logger=logger,
        num_features=cfg.features.num_features,
        cat_features=cfg.features.cat_features,
    )
    test_x = data_loader.load_test_dataset()
    features = [*cfg.features.num_features, *cfg.features.cat_features]
    test_x = test_x[features]

    # load model
    trainer = hydra.utils.instantiate(
        cfg.models,
        logger=logger,
        features=features,
        cat_features=cfg.features.cat_features,
    )
    models = trainer.load_model()

    preds = np.mean([trainer.predict(model, test_x) for model in tqdm(models.values())], axis=0)

    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")
    submit[cfg.data.target] = preds
    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
