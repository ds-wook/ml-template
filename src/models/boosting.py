from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from omegaconf import OmegaConf
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
        split_type: str = "day_of_week",
        logger: logging.Logger = None,
    ) -> None:
        super().__init__(
            model_path,
            results,
            params,
            early_stopping_rounds,
            num_boost_round,
            verbose_eval,
            seed,
            features,
            cat_features,
            n_splits,
            split_type,
            logger,
        )

    def _fit(
        self: Self,
        X_train: pl.DataFrame | np.ndarray,
        y_train: pl.Series | np.ndarray,
        X_valid: pl.DataFrame | np.ndarray | None = None,
        y_valid: pl.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        X_train, y_train = X_train[self.features].to_pandas(), y_train.to_pandas()
        X_valid, y_valid = X_valid[self.features].to_pandas(), y_valid.to_pandas()

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
            feval=self._competition_score_lgb,
            callbacks=callbacks,
        )

        return model

    def _predict(self: Self, X: pl.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(
            lgb.Dataset(
                X[self.features].to_pandas(), categorical_feature=self.cat_features
            )
        )

    def load_model(self: Self) -> lgb.Booster:
        return lgb.Booster(model_file=Path(self.model_path) / f"{self.results}.model")

    def save_model(self: Self, save_dir: Path) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(str(save_dir / f"{self.results}.model"))
