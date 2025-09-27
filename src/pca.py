# src/pca.py
from __future__ import annotations
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def save_pca_variance_plot(X, out_path="results/pca_variance.png", n=10):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(n, Xs.shape[1]))
    pca.fit(Xs)
    evr = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(range(1, len(evr)+1), np.cumsum(evr), marker="o")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA – Cumulative Variance")
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return evr
