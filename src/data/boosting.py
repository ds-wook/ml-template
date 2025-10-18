from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import BaseDataLoader


class BoostingDataLoader(BaseDataLoader):
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
        seed: int = 42,
    ):
        super().__init__(
            logger,
            seed,
            path,
            encoder_path,
            train,
            test,
            submit,
            target,
            cat_features,
            num_features,
        )

    def load_train_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load train dataset
        """
        self.logger.info("Loading train dataset")

        train = pd.read_parquet(Path(self.path) / f"{self.train}.parquet")
        train = self._categorize_train_features(train)
        train = self._numerical_train_scaling(train)
        train_x = train.drop(columns=[self.target])
        train_y = train[self.target]

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        """
        Load test dataset
        """
        self.logger.info("Loading test dataset")

        test = pd.read_parquet(Path(self.path) / f"{self.test}.parquet")
        test = self._categorize_test_features(test)
        test = self._numerical_test_scaling(test)
        test_x = test.drop(columns=[self.target])

        return test_x
