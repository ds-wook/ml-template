from omegaconf import DictConfig

from .base import BaseModel
from .nn import TabNetTrainer
from .tree import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer

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
