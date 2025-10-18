from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from typing_extensions import Self

from utils.metric import get_metrics


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
    def _predict(self: Self, model: Any, X: pd.DataFrame | np.ndarray):
        raise NotImplementedError

    def run_cv_training(
        self: Self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> Self:
        oof_preds = np.zeros(y.shape[0])
        models = {}

        kfold = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        k_splits = kfold.split(X, y)

        with tqdm(k_splits, total=kfold.get_n_splits(X, y)) as pbar:
            for fold, (train_idx, valid_idx) in enumerate(pbar):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = self._predict(model, X_valid)
                models[f"fold_{fold}"] = model
                metrics = get_metrics(y_valid, oof_preds[valid_idx])
                self.logger.info(f"Fold {fold} metrics: {metrics}")

            self.logger.info(f"OOF metrics: {get_metrics(y, oof_preds)}")

        self.result = ModelResult(oof_preds=oof_preds, models=models)
        return self
