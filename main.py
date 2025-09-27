# main.py
from __future__ import annotations
import os
import sys
import glob
import pandas as pd

# local modules
from src.dataset import load_raw, preprocess
from src.train import train_pipeline
from src.evaluate import save_metrics, save_predictions
from src.pca import save_pca_variance_plot

RESULTS_DIR = "results"

def main():
    print(">>> main.py started")

    # Ensure outputs dir exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 👀 Check data/raw/ folder contents
    print("\n=== Checking data/raw/ folder ===")
    files = glob.glob("data/raw/*")
    if not files:
        print("⚠️ No files found in data/raw/")
    else:
        for f in files:
            print(f" - {f} ({os.path.getsize(f)} bytes)")
    print("=================================\n")

    # 1) Load raw train/test tables
    train_df, test_df = load_raw()
    print("Train columns:", list(train_df.columns))
    print("Test  columns:", list(test_df.columns))
    print("\nTrain head:\n", train_df.head(3))

    # 2) Preprocess / feature engineering
    X, y, X_test, feat_names, cat_map = preprocess(train_df, test_df)
    print(f"\nPrepared shapes → X: {X.shape}, y: {None if y is None else y.shape}, X_test: {X_test.shape}")
    print("Features used:", feat_names)

    # 3) PCA variance plot (exploratory only)
    try:
        save_pca_variance_plot(X, out_path=os.path.join(RESULTS_DIR, "pca_variance.png"), n=10)
        print("Saved PCA variance plot → results/pca_variance.png")
    except Exception as e:
        print("PCA plot skipped:", e)

    # 4) Train ANN pipeline + validation metrics
    model, metrics = train_pipeline(X, y)
    print("\n=== Validation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    save_metrics(metrics, out_csv=os.path.join(RESULTS_DIR, "metrics.csv"))

    # 5) Predict for test set
    vidids = test_df["vidid"] if "vidid" in test_df.columns else None
    pred_path = save_predictions(model, X_test, vidids=vidids, out_csv=os.path.join(RESULTS_DIR, "predictions.csv"))
    print(f"\nSaved predictions → {pred_path}")

    # 6) Preview outputs
    try:
        print("\n--- metrics.csv (last row) ---")
        print(pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv")).tail(1))
    except Exception as e:
        print("Could not read metrics.csv:", e)

    try:
        print("\n--- predictions.csv (first 5) ---")
        print(pd.read_csv(os.path.join(RESULTS_DIR, "predictions.csv")).head())
    except Exception as e:
        print("Could not read predictions.csv:", e)

    # 7) List everything in results/
    try:
        print("\nFiles currently in results/:")
        for fn in sorted(os.listdir(RESULTS_DIR)):
            print(" -", fn)
    except Exception as e:
        print("Could not list results/:", e)

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("\n[ERROR] Pipeline failed:", err, file=sys.stderr)
        raise
