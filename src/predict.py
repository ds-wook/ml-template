from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Selected model: {cfg.models.results}")

    # load test dataset
    data_loader = instantiate(cfg.data.loader)
    test_x = data_loader.load_test()

    # load model
    model = instantiate(
        cfg.models,
        logger=logger,
        features=cfg.features.num_features,
        cat_features=cfg.features.cat_features,
        n_splits=cfg.data.n_splits,
        split_type=cfg.data.split_type,
    )
    models = model.load_model()

    # predict
    preds = np.mean([model.predict(test_x) for model in models.values()])

    submit = pd.read_csv(Path(cfg.data.path) / cfg.data.submit)
    submit[cfg.data.target] = preds
    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)
