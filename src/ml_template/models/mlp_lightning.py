import logging
import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from data.mlp import TabularDataset
from models.base import BaseModel


class MLPLightningModule(L.LightningModule):
    """PyTorch Lightning Module for MLP with separate handling for categorical and numerical features"""

    def __init__(
        self,
        num_features_dim: int,
        cat_feature_sizes: list[int],
        embedding_dim: int = 8,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, embedding_dim)
                for num_classes in cat_feature_sizes
            ]
        )

        # Calculate total input dimension
        total_cat_dim = len(cat_feature_sizes) * embedding_dim
        total_input_dim = num_features_dim + total_cat_dim

        # MLP layers
        layers = []
        input_dim = total_input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(input_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: numerical features [batch_size, num_features_dim]
            x_cat: categorical features [batch_size, num_cat_features]
        """
        # Embed categorical features
        cat_embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            cat_embeddings.append(emb_layer(x_cat[:, i].long()))

        # Concatenate all embeddings
        if cat_embeddings:
            cat_embedded = torch.cat(cat_embeddings, dim=1)
            # Concatenate numerical and categorical features
            x = torch.cat([x_num, cat_embedded], dim=1)
        else:
            x = x_num

        return torch.sigmoid(self.mlp(x))

    def training_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        y_hat = self(x_num, x_cat).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        y_hat = self(x_num, x_cat).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class TabularDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for tabular data"""

    def __init__(
        self,
        train_dataset: TabularDataset,
        val_dataset: TabularDataset,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class DeepTrainerLightning(BaseModel):
    def __init__(
        self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        features: list[str],
        cat_features: list[str],
        hidden_dims: list[int] = [128, 64],
        embedding_dim: int = 8,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda",
        seed: int = 42,
        n_splits: int = 5,
        logger: logging.Logger = None,
        cat_feature_sizes: list[int] = None,
        early_stopping_rounds: int = 10,
        verbose_eval: int = 10,
    ) -> None:
        super().__init__(
            model_path=model_path,
            results=results,
            params=params,
            features=features,
            cat_features=cat_features,
            seed=seed,
            n_splits=n_splits,
            logger=logger,
        )
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        # Separate num and cat features
        self.num_features = [f for f in features if f not in cat_features]
        self.cat_feature_sizes = cat_feature_sizes  # Will be set during training

        if self.device == "cpu" and device == "cuda":
            self.logger.warning("CUDA not available, using CPU instead")

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> MLPLightningModule:
        """Train a single model using PyTorch Lightning"""
        X_train = X_train[self.features]
        X_valid = X_valid[self.features]
        y_train = y_train.astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        # Calculate categorical feature sizes (vocabulary size for each cat feature)
        if not self.cat_feature_sizes:
            self.cat_feature_sizes = [
                int(X_train[col].max()) + 1 for col in self.cat_features
            ]
            self.logger.info(f"Categorical feature sizes: {self.cat_feature_sizes}")
            self.logger.info(
                f"Num features: {len(self.num_features)}, Cat features: {len(self.cat_features)}"
            )

        # Create datasets
        train_dataset = TabularDataset(
            X_train, y_train, self.num_features, self.cat_features
        )
        valid_dataset = TabularDataset(
            X_valid, y_valid, self.num_features, self.cat_features
        )

        # Create data module
        data_module = TabularDataModule(
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            batch_size=self.batch_size,
        )

        # Create model
        num_features_dim = len(self.num_features)
        model = MLPLightningModule(
            num_features_dim=num_features_dim,
            cat_feature_sizes=self.cat_feature_sizes,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            lr=self.lr,
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_rounds,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best_model",
            ),
        ]

        # Setup logger
        csv_logger = CSVLogger(
            save_dir=Path(self.model_path) / f"{self.results}",
            name="logs",
        )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            logger=csv_logger,
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        # Train the model
        trainer.fit(model, data_module)

        # Load the best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            model = MLPLightningModule.load_from_checkpoint(
                best_model_path,
                num_features_dim=num_features_dim,
                cat_feature_sizes=self.cat_feature_sizes,
                embedding_dim=self.embedding_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                lr=self.lr,
            )
            self.logger.info(f"Loaded best model from {best_model_path}")

        return model

    def _predict(
        self: Self, model: MLPLightningModule, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Make predictions with a trained model"""
        X = X[self.features]
        dataset = TabularDataset(
            X, num_features=self.num_features, cat_features=self.cat_features
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                # DataLoader returns a list when dataset returns tuple
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x_num, x_cat = batch[0], batch[1]
                    x_num = x_num.to(self.device)
                    x_cat = x_cat.to(self.device)
                    pred = model(x_num, x_cat).squeeze()
                else:
                    x_num = batch.to(self.device)
                    pred = model(x_num).squeeze()
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds).flatten()

    def save_model(self: Self, save_dir: Path) -> None:
        """Save trained models"""
        if not os.path.exists(save_dir / f"{self.results}"):
            os.makedirs(save_dir / f"{self.results}", exist_ok=True)

        if self.result.models is not None:
            for fold, model in tqdm(self.result.models.items(), desc="Saving models"):
                torch.save(
                    model.state_dict(),
                    save_dir / f"{self.results}" / f"{fold}.pth",
                )
        else:
            torch.save(
                self.model.state_dict(),
                save_dir / f"{self.results}" / f"{self.results}.pth",
            )

    def load_model(self: Self) -> dict[str, MLPLightningModule] | MLPLightningModule:
        """Load trained models"""
        num_features_dim = len(self.num_features)

        if self.n_splits > 1:
            models = {}
            for model_file in os.listdir(Path(self.model_path) / f"{self.results}"):
                if model_file.endswith(".pth"):
                    model = MLPLightningModule(
                        num_features_dim=num_features_dim,
                        cat_feature_sizes=self.cat_feature_sizes,
                        embedding_dim=self.embedding_dim,
                        hidden_dims=self.hidden_dims,
                        dropout=self.dropout,
                        lr=self.lr,
                    )
                    model.load_state_dict(
                        torch.load(
                            Path(self.model_path) / f"{self.results}" / model_file,
                            map_location=self.device,
                        )
                    )
                    model.eval()
                    models[model_file.replace(".pth", "")] = model
            return models
        else:
            model = MLPLightningModule(
                num_features_dim=num_features_dim,
                cat_feature_sizes=self.cat_feature_sizes,
                embedding_dim=self.embedding_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                lr=self.lr,
            )
            model.load_state_dict(
                torch.load(
                    Path(self.model_path) / f"{self.results}.pth",
                    map_location=self.device,
                )
            )
            model.eval()
            return model

    def predict(
        self: Self, model: MLPLightningModule, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Make predictions with a trained model"""
        return self._predict(model, X)
