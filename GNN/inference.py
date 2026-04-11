"""
GraGOD — Real-World Inference Script
========================================================================
Runs anomaly detection on new, unlabeled data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.real_dataset import SENSOR_COLS
from data.dataset import TimeSeriesDataset
from model.gdn import GDN

def run_inference(csv_path: str, artifact_path: str = "model_checkpoints/gdn_artifacts.pt"):
    print("=" * 60)
    print("  GraGOD — Inference Mode (Unlabeled Data)")
    print("=" * 60)

    # ── 1. Load Artifacts ────────────────────────────────────────────────
    print("[1/4] Loading model artifacts...")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"\n✗ Artifacts not found at {artifact_path}\n  Please run train.py first to generate the model.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = torch.load(artifact_path, map_location=device, weights_only=False)

    cfg       = artifacts["cfg"]
    mu        = artifacts["mu"]
    sigma     = artifacts["sigma"]
    threshold = artifacts["threshold"]

    print(f"  ✓ Device          : {device}")
    print(f"  ✓ Fixed Threshold : {threshold:.4f}")

    # ── 2. Load New Unlabeled Data ───────────────────────────────────────
    print(f"\n[2/4] Loading new data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"✗ CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    missing_cols = [col for col in SENSOR_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required sensor columns in CSV: {missing_cols}")

    # We don't need the Occupancy label, only the raw sensors
    feature_df = df[SENSOR_COLS].copy()
    data_arr = feature_df.values.astype(np.float32)
    print(f"  ✓ Loaded {len(data_arr)} rows.")

    # ── 3. Build Dataset & Model ─────────────────────────────────────────
    print("\n[3/4] Preparing sliding windows and initializing model...")
    # Note: We pass the saved mu and sigma so it scales exactly like the training data
    dataset = TimeSeriesDataset(data_arr, window_size=cfg["window_size"], normalize=True, mu=mu, sigma=sigma)
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = GDN(
        n_sensors     = len(SENSOR_COLS),
        window_size   = cfg["window_size"],
        embed_dim     = cfg["embed_dim"],
        hidden_dim    = cfg["hidden_dim"],
        top_k         = cfg["top_k"],
        dynamic_graph = cfg["dynamic_graph"],
    )
    model.load_state_dict(artifacts["model_state"])
    model.to(device)
    model.eval()

    # ── 4. Run Inference ─────────────────────────────────────────────────
    print("\n[4/4] Calculating deviations...")
    all_errors = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            # The error is the difference between predicted reality and actual reality
            errors = torch.abs(preds - y) 
            all_errors.append(errors.cpu().numpy())

    # Shape: (Total Windows, Num Sensors)
    per_sensor_errors = np.concatenate(all_errors, axis=0)

    # Combine per-sensor errors into a single score using configured method (mean/max)
    reduce_method = cfg.get("score_reduce", "mean")
    if reduce_method == "mean":
        scores = per_sensor_errors.mean(axis=1)
    else:
        scores = per_sensor_errors.max(axis=1)

    # Flag anomalies based on the strict threshold learned during training
    anomalies = (scores > threshold).astype(int)
    num_anomalies = anomalies.sum()

    # ── 5. Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INFERENCE RESULTS")
    print("=" * 60)
    print(f"  Total Windows Processed : {len(scores)}")
    print(f"  Anomalies Detected      : {num_anomalies} ({(num_anomalies/len(scores))*100:.2f}%)")

    if num_anomalies > 0:
        print("\n  Top 5 Most Severe Anomalies:")
        # Find indices where anomaly == 1
        anomaly_indices = np.where(anomalies == 1)[0]
        # Sort these indices by how high the score was
        sorted_anomaly_idx = anomaly_indices[np.argsort(scores[anomaly_indices])[::-1]]

        for i in range(min(5, len(sorted_anomaly_idx))):
            idx = sorted_anomaly_idx[i]
            original_row_idx = idx + cfg["window_size"]
            print(f"    - CSV Row {original_row_idx:05d}  |  Score = {scores[idx]:.4f}  (Threshold: {threshold:.4f})")

        # Fault Localization for the absolute worst anomaly
        worst_idx = sorted_anomaly_idx[0]
        worst_sensor_errors = per_sensor_errors[worst_idx]
        ranked_sensors = np.argsort(worst_sensor_errors)[::-1]
        
        print(f"\n  Fault Localization for Worst Anomaly (Row {worst_idx + cfg['window_size']}):")
        for rank, s_idx in enumerate(ranked_sensors, 1):
            print(f"    {rank}. {SENSOR_COLS[s_idx]:<15s} error={worst_sensor_errors[s_idx]:.5f}")

    print("\n✓ Inference complete.")
    
    # Return arrays in case you want to plot them later
    return scores, anomalies

if __name__ == "__main__":
    # You can point this to any new CSV file containing the 5 sensor columns
    TARGET_CSV = "data/sensor_data.csv" 
    run_inference(TARGET_CSV)