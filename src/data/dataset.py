# This code is a Python file that defines a function for preprocessing and loading datasets.
# The function reads in the dataset, performs necessary preprocessing tasks,
# and transforms it into a format that can be used for model training.
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from generator.featurize import FeatureEngineer


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    return


def load_test_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    return
