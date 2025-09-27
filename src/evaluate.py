# src/evaluate.py
from __future__ import annotations
import os
import csv
import numpy as np
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_metrics(metrics: dict, out_csv="results/metrics.csv"):
    ensure_dir(os.path.dirname(out_csv))
    # append or create
    mode = "a" if os.path.exists(out_csv) else "w"
    with open(out_csv, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["rmse","mae","r2"])
        w.writerow([metrics["rmse"], metrics["mae"], metrics["r2"]])

def save_predictions(model, X_test, vidids=None, out_csv="results/predictions.csv"):
    ensure_dir(os.path.dirname(out_csv))
    preds = model.predict(X_test)
    df = pd.DataFrame({"prediction": preds})
    if vidids is not None:
        df.insert(0, "vidid", vidids)
    df.to_csv(out_csv, index=False)
    return out_csv