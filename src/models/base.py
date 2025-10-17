from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import Self


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(ABC):
    def __init__(
        self: Self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        features: list[str],
        cat_features: list[str],
        early_stopping_rounds: int = 100,
        num_boost_round: int = 1000,
        verbose_eval: int = 100,
        seed: int = 42,
        n_splits: int = 5,
        logger: logging.Logger = None,
    ) -> None:
        self.model_path = model_path
        self.results = results
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.seed = seed
        self.model = None
        self.features = features
        self.cat_features = cat_features
        self.n_splits = n_splits
        self.logger = logger

    @abstractmethod
    def save_model(self, save_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self: Self):
        # return model
        raise NotImplementedError

    @abstractmethod
    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    @abstractmethod
    def _predict(self: Self, X: pd.DataFrame | np.ndarray):
        raise NotImplementedError

    def run_cv_training(
        self: Self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> Self:
        oof_preds = np.zeros(X.shape[0])
        models = {}

        with tqdm(self.n_splits, total=self.n_splits) as pbar:
            for fold, (train_idx, valid_idx) in enumerate(pbar, 1):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                if "xgboost" in self.results:
                    X_train, X_valid = self._encode_categorical_count(X_train, X_valid)

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = self._predict(model, X_valid)
                models[f"fold_{fold}"] = model
                del X_train, X_valid, y_train, y_valid, model
                gc.collect()

        self.result = ModelResult(oof_preds=oof_preds, models=models)

        return self
