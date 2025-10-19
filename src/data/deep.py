from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BaseDataLoader


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data"""

    def __init__(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray = None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class DeepDataLoader(BaseDataLoader):
    def __init__(
        self,
        logger: logging.Logger,
        path: str,
        encoder_path: str,
        train: str,
        test: str,
        submit: str,
        target: str,
        cat_features: list[str],
        num_features: list[str],
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__(
            logger=logger,
            path=path,
            encoder_path=encoder_path,
            train=train,
            test=test,
            submit=submit,
            target=target,
            cat_features=cat_features,
            num_features=num_features,
            seed=seed,
        )
        self.batch_size = batch_size

    def load_train_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load train dataset for deep learning
        Returns raw dataframes (conversion to DataLoader happens in model trainer)
        """
        self.logger.info("Loading train dataset for deep learning")

        train = pd.read_csv(Path(self.path) / f"{self.train}.csv")
        train = self._categorize_train_features(train)
        train = self._numerical_train_scaling(train)
        train = train.fillna(0)
        train_x = train.drop(columns=[self.target])
        train_y = train[self.target].astype(int)  # Convert boolean to int for BCE loss

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        """
        Load test dataset for deep learning
        """
        self.logger.info("Loading test dataset for deep learning")

        test = pd.read_csv(Path(self.path) / f"{self.test}.csv")
        test = self._categorize_test_features(test)
        test = self._numerical_test_scaling(test)
        test = test.fillna(0)

        return test

    def create_dataloader(
        self, X: pd.DataFrame, y: pd.Series = None, shuffle: bool = True
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from pandas DataFrame
        """
        dataset = TabularDataset(X, y)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )
