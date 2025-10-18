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

    def load_train_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load train dataset
        """
        print(self.path, self.train)
        self.logger.info("Loading train dataset")

        train = pd.read_csv(Path(self.path) / f"{self.train}.csv")
        train = self._categorize_train_features(train)
        train_x = train.drop(columns=[self.target])
        train_y = train[self.target]

        return train_x, train_y

    def load_test_dataset(self) -> pd.DataFrame:
        """
        Load test dataset
        """
        self.logger.info("Loading test dataset")

        test = pd.read_csv(Path(self.path) / f"{self.test}.csv")
        test = self._categorize_test_features(test)
        test_x = test.drop(columns=[self.target])

        return test_x
