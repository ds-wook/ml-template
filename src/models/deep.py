import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from data.deep import TabularDataset
from models.base import BaseModel


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepTrainer(BaseModel):
    def __init__(
        self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        features: list[str],
        cat_features: list[str],
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda",
        seed: int = 42,
        n_splits: int = 5,
        logger: logging.Logger = None,
    ):
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
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"

        if self.device == "cpu" and device == "cuda":
            self.logger.warning("CUDA not available, using CPU instead")

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> nn.Module:
        """Train a single model"""
        X_train = X_train[self.features]
        X_valid = X_valid[self.features]
        y_train = y_train.astype(np.int8)
        y_valid = y_valid.astype(np.int8)

        # Create model
        input_dim = len(self.features)
        model = MLP(input_dim, self.hidden_dims, self.dropout).to(self.device)

        # Create dataloaders
        train_dataset = TabularDataset(X_train, y_train)
        valid_dataset = TabularDataset(X_valid, y_valid)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).squeeze()
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x).squeeze()
                    loss = criterion(pred, y)
                    val_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(valid_loader)
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

        return model

    def _predict(
        self: Self, model: nn.Module, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Make predictions with a trained model"""
        X = X[self.features]
        dataset = TabularDataset(X)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        model.eval()
        preds = []
        with torch.no_grad():
            for x in loader:
                if isinstance(x, tuple):
                    x = x[0]
                x = x.to(self.device)
                pred = model(x).squeeze()
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

    def load_model(self: Self) -> dict[str, nn.Module] | nn.Module:
        """Load trained models"""
        input_dim = len(self.features)

        if self.n_splits > 1:
            models = {}
            for model_file in os.listdir(Path(self.model_path) / f"{self.results}"):
                if model_file.endswith(".pth"):
                    model = MLP(input_dim, self.hidden_dims, self.dropout).to(
                        self.device
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
            model = MLP(input_dim, self.hidden_dims, self.dropout).to(self.device)
            model.load_state_dict(
                torch.load(
                    Path(self.model_path) / f"{self.results}.pth",
                    map_location=self.device,
                )
            )
            model.eval()
            return model
