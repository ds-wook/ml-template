from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing_extensions import Self

from .nn.tabnet import TabNetTrainer
from .tree.boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer

BulidModel = CatBoostTrainer | LightGBMTrainer | XGBoostTrainer | TabNetTrainer


def bulid_model(cfg: DictConfig) -> BulidModel:
    model_type = {
        "lightgbm": LightGBMTrainer(cfg),
        "xgboost": XGBoostTrainer(cfg),
        "catboost": CatBoostTrainer(cfg),
        "tabnet": TabNetTrainer(cfg),
    }

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def save_model(self, save_dir: Path) -> None:
        joblib.dump(self.result, save_dir)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> Any:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(model, xgb.Booster):
            return model.predict(xgb.DMatrix(X))

        elif isinstance(model, TabNetRegressor):
            return model.predict(X.to_numpy()).reshape(-1)

        else:
            return model.predict(X)

    def run_cv_training(self, X: pd.DataFrame, y: pd.Series) -> Self:
        oof_preds = np.zeros(X.shape[0])
        models = {}
        kfold = KFold(n_splits=self.cfg.data.n_splits, shuffle=True, random_state=self.cfg.data.seed)

        with tqdm(kfold.split(X=X), total=self.cfg.data.n_splits, desc="CV", leave=False) as pbar:
            for fold, (train_idx, valid_idx) in enumerate(iterable=pbar, start=1):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = self._predict(model, X_valid)

                models[f"fold_{fold}"] = model

            del X_train, X_valid, y_train, y_valid
            gc.collect()

        print(f"CV Score: {mean_absolute_error(y, oof_preds):.6f}")

        self.result = ModelResult(oof_preds=oof_preds, models=models)

        return self
