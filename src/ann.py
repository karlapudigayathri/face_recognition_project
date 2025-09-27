# src/ann.py
from __future__ import annotations
from sklearn.neural_network import MLPRegressor

def build_mlp(hidden=(128, 64), random_state: int = 42) -> MLPRegressor:
    """
    Simple ANN for regression.
    - ReLU hidden layers, 'adam' solver
    - Small L2 alpha to reduce overfitting
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=400,
        random_state=random_state,
        verbose=False
    )
