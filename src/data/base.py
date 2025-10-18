from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer


class BaseDataLoader(ABC):
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
        self.logger = logger
        self.seed = seed
        self.path = path
        self.encoder_path = encoder_path
        self.train = train
        self.test = test
        self.submit = submit
        self.target = target
        self.num_features = num_features
        self.cat_features = cat_features

    def _categorize_train_features(self, train_x: pd.DataFrame) -> pd.DataFrame:
        le = LabelEncoder()
        train_x[[*self.cat_features]] = le.fit_transform(train_x[[*self.cat_features]])
        joblib.dump(le, Path(self.encoder_path) / "label_encoder.pkl")

        return train_x

    def _categorize_test_features(self, test_x: pd.DataFrame) -> pd.DataFrame:
        """Categorical encoding for test data
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        le = joblib.load(Path(self.encoder_path) / "label_encoder.pkl")
        test_x[[*self.cat_features]] = le.transform(test_x[[*self.cat_features]])

        return test_x

    def _numerical_train_scaling(self, train: pd.DataFrame) -> pd.DataFrame:
        scaler = QuantileTransformer(n_quantiles=100, output_distribution="normal")
        train[[*self.num_features]] = scaler.fit_transform(train[[*self.num_features]])
        joblib.dump(scaler, Path(self.encoder_path) / "rankgauss.pkl")

        return train

    def _numerical_test_scaling(self, test: pd.DataFrame) -> pd.DataFrame:
        scaler = joblib.load(Path(self.encoder_path) / "rankgauss.pkl")
        test[[*self.num_features]] = scaler.transform(test[[*self.num_features]])

        return test

    @abstractmethod
    def load_train_dataset(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_test_dataset(self) -> pd.DataFrame:
        raise NotImplementedError
