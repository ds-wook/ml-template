from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from typing_extensions import Self

from models.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(
        self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        early_stopping_rounds: int,
        num_boost_round: int,
        verbose_eval: int,
        seed: int,
        features: list[str],
        cat_features: list[str],
        n_splits: int = 5,
        logger: logging.Logger = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            results=results,
            params=params,
            features=features,
            cat_features=cat_features,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            seed=seed,
            n_splits=n_splits,
            logger=logger,
        )

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        X_train = X_train[self.features]
        X_valid = X_valid[self.features]
        # set params
        params = OmegaConf.to_container(self.params)
        params["seed"] = self.seed

        train_set = lgb.Dataset(
            X_train,
            y_train,
            params=params,
            categorical_feature=self.cat_features,
            feature_name=self.features,
        )
        valid_set = lgb.Dataset(
            X_valid,
            y_valid,
            params=params,
            categorical_feature=self.cat_features,
            feature_name=self.features,
        )

        # dart boosting의 경우 early_stopping 사용하지 않음 (내부적으로 처리불가하여 콜백으로 처리)
        callbacks = (
            [
                lgb.log_evaluation(self.verbose_eval),
                lgb.early_stopping(self.early_stopping_rounds),
            ]
            if params.get("boosting_type") != "dart"
            else [lgb.log_evaluation(self.verbose_eval)]
        )

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[valid_set],
            num_boost_round=self.num_boost_round,
            callbacks=callbacks,
        )

        return model

    def _predict(
        self: Self, model: lgb.Booster, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        return model.predict(X[self.features])

    def predict(
        self: Self, model: lgb.Booster, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        return self._predict(model, X)

    def load_model(self: Self) -> dict[str, lgb.Booster] | lgb.Booster:
        if self.n_splits > 1:
            models = {}
            for model_file in os.listdir(Path(self.model_path) / f"{self.results}"):
                models[model_file] = lgb.Booster(
                    model_file=str(
                        Path(self.model_path) / f"{self.results}" / model_file
                    )
                )
            return models
        else:
            return lgb.Booster(
                model_file=str(Path(self.model_path) / f"{self.results}.model")
            )

    def save_model(self: Self, save_dir: Path) -> None:
        if not os.path.exists(save_dir / f"{self.results}"):
            os.makedirs(save_dir / f"{self.results}", exist_ok=True)

        if self.result.models is not None:
            for fold, model in tqdm(self.result.models.items(), desc="Saving models"):
                model.save_model(save_dir / f"{self.results}" / f"{fold}.model")
        else:
            self.model.save_model(
                save_dir / f"{self.results}" / f"{self.results}.model"
            )
