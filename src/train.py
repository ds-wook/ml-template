from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig

from data.dataset import load_train_dataset
from models.nn import TabNetTrainer
from models.tree import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        save_path = Path(cfg.models.path)

        train_x, train_y = load_train_dataset(cfg)
        train_x = reduce_mem_usage(train_x)

        if cfg.models.name == "lightgbm":
            # train model
            lgb_trainer = LightGBMTrainer(cfg)
            lgb_trainer.run_cv_training(train_x, train_y)

            # save model
            lgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "catboost":
            # train model
            cb_trainer = CatBoostTrainer(cfg)
            cb_trainer.run_cv_training(train_x, train_y)

            # save model
            cb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "xgboost":
            # train model
            xgb_trainer = XGBoostTrainer(cfg)
            xgb_trainer.run_cv_training(train_x, train_y)

            # save model
            xgb_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        elif cfg.models.name == "tabnet":
            train_x = train_x.fillna(0)
            # train model
            tabnet_trainer = TabNetTrainer(cfg)
            tabnet_trainer.run_cv_training(train_x, train_y)

            # save model
            tabnet_trainer.save_model(save_path / f"{cfg.models.results}.pkl")

        else:
            raise NotImplementedError


if __name__ == "__main__":
    _main()
