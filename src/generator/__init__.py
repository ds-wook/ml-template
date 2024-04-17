# This code represents a class that performs data preprocessing.
# Data preprocessing is the process of preparing data for machine learning model training, refining and transforming data into a format that is easy for the model to learn.
# The class carries out various preprocessing tasks, such as encoding categorical variables, to facilitate the learning process for the machine learning model.
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from .featurize import FeatureEngineer


class BaseFeatureEngineer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _categorize_train_features(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.cfg.data.encoder

        for cat_feature in tqdm(self.cfg.data.categorical_features, desc="Encoding train data", leave=False):
            le = LabelEncoder()
            train[cat_feature] = le.fit_transform(train[cat_feature])

            joblib.dump(le, path / f"{cat_feature}.pkl")

        return train

    def _categorize_test_features(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.cfg.data.encoder

        for cat_feature in tqdm(self.cfg.data.categorical_features, desc="Encoding test data", leave=False):
            le = joblib.load(path / f"{cat_feature}.pkl")
            test[cat_feature] = le.transform(test[cat_feature].astype(str))

        return test

    def _standard_train_features(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            train: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.cfg.data.encoder

        for num_feature in tqdm(self.cfg.data.numerical_features, desc="Encoding train data", leave=False):
            scaler = StandardScaler()
            train[num_feature] = scaler.fit_transform(train[num_feature].to_numpy().reshape(-1, 1))

            joblib.dump(scaler, path / f"{num_feature}.pkl")

        return train

    def _standard_test_features(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Categorical encoding
        Args:
            config: config
            test: dataframe
        Returns:
            dataframe
        """

        path = Path(get_original_cwd()) / self.cfg.data.encoder

        for num_feature in tqdm(self.cfg.data.numerical_features, desc="Encoding test data", leave=False):
            scaler = joblib.load(path / f"{num_feature}.pkl")
            test[num_feature] = scaler.transform(test[num_feature].to_numpy().reshape(-1, 1))

        return test
