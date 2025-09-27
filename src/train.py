# src/train.py
from __future__ import annotations
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from .ann import build_mlp

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_pipeline(X, y):
    """
    Create a pipeline: Standardize → MLPRegressor.
    Returns the fitted pipeline and a dict with validation metrics.
    """
    X_tr, X_val, y_tr, y_val = train_val_split(X, y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", build_mlp())
    ])
    pipe.fit(X_tr, y_tr)

    # Validation metrics
    y_pred = pipe.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae  = float(mean_absolute_error(y_val, y_pred))
    r2   = float(r2_score(y_val, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    return pipe, metrics