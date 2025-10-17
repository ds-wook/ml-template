import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import BaseModel


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


class DeepModel(BaseModel):
    def __init__(
        self, model, lr=1e-3, epochs=10, device="cuda", logger: logging.Logger = None
    ):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.logger = logger

    def fit(self, train_dl, val_dl=None):
        criterion = nn.BCELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.model.train()
            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x).squeeze()
                loss = criterion(pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
        return self

    def _predict(self, X: pd.DataFrame | np.ndarray):
        model = self.model.eval()
        with torch.no_grad():
            preds = model(X).cpu().numpy().flatten()
        return preds.flatten()
