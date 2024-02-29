from __future__ import annotations

import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_tabnet.tab_model import TabNetRegressor

from models.base import BaseModel


class EarlyStoppingCallback:
    def __init__(self, min_delta: float = 0.1, patience: int = 5) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.best_epoch_score = 0

        self.attempt = 0
        self.best_score = None
        self.stop_training = False

    def __call__(self, validation_loss: float) -> None:
        self.epoch_score = validation_loss

        if self.best_epoch_score == 0:
            self.best_epoch_score = self.epoch_score

        elif self.epoch_score > self.best_epoch_score - self.min_delta:
            self.attempt += 1

            if self.attempt >= self.patience:
                self.stop_training = True

        else:
            self.best_epoch_score = self.epoch_score
            self.attempt = 0


class TabNetTrainer(BaseModel):
    def __init__(self, cfg: DictConfig, cat_idxs: list[int] = [], cat_dims: list[int] = []) -> None:
        super().__init__(cfg)
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> TabNetRegressor:
        """method train"""
        model = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.cfg.models.params.lr),
            scheduler_params={
                "step_size": self.cfg.models.params.step_size,
                "gamma": self.cfg.models.params.gamma,
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=self.cfg.models.params.mask_type,
            n_steps=self.cfg.models.params.n_steps,
            n_d=self.cfg.models.params.n_d,
            n_a=self.cfg.models.params.n_a,
            lambda_sparse=self.cfg.models.params.lambda_sparse,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            verbose=self.cfg.models.params.verbose,
        )

        model.fit(
            X_train=X_train.to_numpy(),
            y_train=y_train.to_numpy().reshape(-1, 1),
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)),
                (X_valid.to_numpy(), y_valid.to_numpy().reshape(-1, 1)),
            ],
            eval_name=[*self.cfg.models.eval_name],
            eval_metric=[*self.cfg.models.eval_metric],
            max_epochs=self.cfg.models.params.max_epochs,
            patience=self.cfg.models.params.patience,
            batch_size=self.cfg.models.params.batch_size,
            virtual_batch_size=self.cfg.models.params.virtual_batch_size,
            num_workers=self.cfg.models.params.num_workers,
            drop_last=False,
        )

        return model
