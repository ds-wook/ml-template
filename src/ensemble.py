from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from models.base import ModelResult


def inference_models(result: list[ModelResult], test_x: pd.DataFrame) -> np.ndarray:
    """
    Given a model, predict probabilities for each class.
    Args:
        model_results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """

    folds = len(result.models)
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(result.models.values(), total=folds, desc="Predicting models"):
        preds += (
            model.predict(xgb.DMatrix(test_x)) / folds
            if isinstance(model, xgb.Booster)
            else model.predict(test_x.to_numpy()) / folds
            if isinstance(model, TabNetRegressor)
            else model.predict(test_x) / folds
        )

    return preds


def get_score(weights: np.ndarray, train_idx: list[int], oofs: list[np.ndarray], preds: np.ndarray) -> float:
    blending = np.zeros_like(oofs[0][train_idx])

    for oof, weight in zip(oofs[:-1], weights):
        blending += weight * oof[train_idx]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx]

    scores = mean_absolute_error(preds[train_idx], blending)

    return scores


def get_best_weights(oofs: np.ndarray, preds: np.ndarray) -> float:
    weights = np.array([1 / len(oofs) for _ in range(len(oofs) - 1)])
    weight_list = []

    kf = KFold(n_splits=5)
    for fold, (train_idx, _) in enumerate(kf.split(oofs[0]), 1):
        res = minimize(get_score, weights, args=(train_idx, oofs, preds), method="Nelder-Mead", tol=1e-6)
        print(f"fold: {fold} res.x: {res.x}")
        weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    mean_weight = np.insert(mean_weight, len(mean_weight), 1 - np.sum(mean_weight))
    print(f"optimized weight: {mean_weight}\n")

    return mean_weight
