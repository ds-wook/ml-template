from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    # load test dataset
    data_loader = instantiate(cfg.data.loader)
    test_x = data_loader.load_test()
    test_x = test_x.to_pandas()

    # load model
    trainer = instantiate(cfg.models)

    preds = trainer.predict(test_x)
