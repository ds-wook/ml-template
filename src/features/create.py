from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from features import BaseFeatureEngineer


class FeatureEngineer(BaseFeatureEngineer):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        super().__init__(cfg)
        df = self._add_time_features(df)
        df = self._add_basic_features(df)
        self.df = df

    def get_train_pipeline(self):
        """
        Get train pipeline
        Returns:
            dataframe
        """

        if self.cfg.models.name == "tabnet":
            self.df = self._standard_train_features(self.df)

        return self.df

    def get_test_pipeline(self):
        """
        Get test pipeline
        Returns:
            dataframe
        """

        if self.cfg.models.name == "tabnet":
            self.df = self._standard_test_features(self.df)

        return self.df
