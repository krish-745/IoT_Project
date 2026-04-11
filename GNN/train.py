"""
GraGOD — Real-Time Streaming Anomaly Detection
========================================================================
REAL DATA MODE: reads from data/sensor_data.csv

This script has been modified to:
  1. Train on the first 80% of the dataset chronologically.
  2. Simulate a real-time incoming data stream on the remaining 20%.
  3. Predict anomalies strictly "on the spot" as each new reading arrives.
  4. Save trained model artifacts so retraining is not needed every time.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import torch
import numpy as np
import pandas as pd

from data.real_dataset import SENSOR_COLS, LABEL_COL
from data.dataset      import TimeSeriesDataset
from model.gdn         import GDN
from utils.trainer     import train, compute_train_errors
from utils.evaluate    import fit_threshold

# ── configuration ─────────────────────────────────────────────────────────────

ARTIFACT_PATH = Path(__file__).parent / "model_checkpoints" / "gdn_artifacts.pt"

CFG = {
    # data
    "csv_path":    Path(__file__).parent / "data" / "sensor_data.csv",
    "window_size": 15,
    "train_split": 0.8,          

    # model — 5 sensor features
    "embed_dim":    32,
    "hidden_dim":   32,
    "top_k":        3,           
    "dynamic_graph": True,       # ON: Let the model learn the sensor relationships

    # training
    "n_epochs":   50,
    "batch_size": 64,
    "lr":         1e-3,
    "patience":   8,

    # anomaly scoring
    "threshold_k":  0.4,
    "score_reduce": "max",

    # misc
    "seed":   42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_or_train_model():
    """
    Load a pre-trained model from artifacts if available,
    otherwise train from scratch and save the artifacts.
    
    Returns: (model, threshold, mu, sigma, cfg)
    """
    n_sensors = len(SENSOR_COLS)
    device = CFG["device"]
    
    # Check if artifacts already exist
    if ARTIFACT_PATH.exists():
        print("\n[INFO] Found pre-trained model artifacts. Loading...")
        artifacts = torch.load(ARTIFACT_PATH, map_location=device, weights_only=False)
        
        cfg       = artifacts["cfg"]
        mu        = artifacts["mu"]
        sigma     = artifacts["sigma"]
        threshold = artifacts["threshold"]
        
        model = GDN(
            n_sensors     = n_sensors,
            window_size   = cfg["window_size"],
            embed_dim     = cfg["embed_dim"],
            hidden_dim    = cfg["hidden_dim"],
            top_k         = cfg["top_k"],
            dynamic_graph = cfg["dynamic_graph"],
        ).to(device)
        model.load_state_dict(artifacts["model_state"])
        model.eval()
        
        print(f"  ✓ Model loaded successfully from {ARTIFACT_PATH}")
        print(f"  ✓ Threshold: {threshold:.4f}")
        return model, threshold, mu, sigma, cfg
    
    # If no artifacts, train from scratch
    print("\n[INFO] No pre-trained model found. Training from scratch...")
    return train_model()


def train_model():
    """Train the model from scratch, save artifacts, and return."""
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    
    n_sensors = len(SENSOR_COLS)
    device = CFG["device"]
    
    # ── 1. Load Data & Split 80/20 ────────────────────────────────────────
    print("\n[1/3] Loading and splitting sensor_data.csv...")
    if not CFG["csv_path"].exists():
        raise FileNotFoundError(f"\n✗ CSV not found at: {CFG['csv_path']}")

    df = pd.read_csv(CFG["csv_path"])
    split_idx = int(CFG["train_split"] * len(df))
    
    train_data = df.iloc[:split_idx].copy()
    
    print(f"  ✓ Total Rows   : {len(df)}")
    print(f"  ✓ Train (80%)  : {len(train_data)} rows")

    # ── 2. Prepare Training Data ──────────────────────────────────────────
    print("\n[2/3] Building training dataset (Normal rows only)...")
    train_normal = train_data[train_data[LABEL_COL] == 1].reset_index(drop=True)
    train_arr = train_normal[SENSOR_COLS].values.astype(np.float32)
    
    train_ds = TimeSeriesDataset(
        train_arr, 
        window_size=CFG["window_size"], 
        normalize=True
    )
    print(f"  ✓ Training samples generated: {len(train_ds)}")

    # ── 3. Initialize & Train Model ───────────────────────────────────────
    print("\n[3/3] Initializing and training GDN model...")
    model = GDN(
        n_sensors     = n_sensors,
        window_size   = CFG["window_size"],
        embed_dim     = CFG["embed_dim"],
        hidden_dim    = CFG["hidden_dim"],
        top_k         = CFG["top_k"],
        dynamic_graph = CFG["dynamic_graph"],
    ).to(device)

    train(
        model         = model,
        train_dataset = train_ds,
        n_epochs      = CFG["n_epochs"],
        batch_size    = CFG["batch_size"],
        lr            = CFG["lr"],
        patience      = CFG["patience"],
        device        = device,
    )

    print("\n  Calibrating threshold...")
    train_errors_tensor = compute_train_errors(model, train_ds, device=device)
    threshold = fit_threshold(train_errors_tensor.numpy(), k=CFG["threshold_k"], reduce=CFG["score_reduce"])
    
    mu = train_ds.mu
    sigma = train_ds.sigma

    # ── Save Artifacts ────────────────────────────────────────────────────
    print(f"\n  Saving model artifacts to {ARTIFACT_PATH}...")
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cfg":         CFG,
        "model_state": model.state_dict(),
        "mu":          mu,
        "sigma":       sigma,
        "threshold":   threshold,
    }, ARTIFACT_PATH)
    print(f"  ✓ Artifacts saved successfully!")
    
    return model, threshold, mu, sigma, CFG


def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 60)
    print("  GraGOD — Real-Time GNN Anomaly Detection")
    print("=" * 60)
    print(f"  Device   : {CFG['device']}")
    print(f"  CSV      : {CFG['csv_path']}")
    
    # ── Load or Train Model ───────────────────────────────────────────────
    model, threshold, mu, sigma, cfg = load_or_train_model()
    
    # ── Load Stream Data ──────────────────────────────────────────────────
    df = pd.read_csv(CFG["csv_path"])
    split_idx = int(CFG["train_split"] * len(df))
    stream_data = df.iloc[split_idx:].reset_index(drop=True)
    print(f"\n  Stream (20%) : {len(stream_data)} rows waiting for real-time feed")
    
    # ── Real-Time Streaming Simulation ────────────────────────────────────
    print("\n" + "=" * 60)
    print(" 🚀 INITIATING REAL-TIME DATA STREAM (Remaining 20%)")
    print("=" * 60)
    
    # Extract normalization parameters
    mu_flat = mu.flatten() if hasattr(mu, 'flatten') else mu
    sigma_flat = sigma.flatten() if hasattr(sigma, 'flatten') else sigma
    
    history_buffer = []
    
    # Metrics tracking
    total_predictions = 0
    correct_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    model.eval()
    device = CFG["device"]
    with torch.no_grad():
        for i, row in stream_data.iterrows():
            features = row[SENSOR_COLS].values.astype(np.float32)
            actual_label = int(row[LABEL_COL])
            truth_is_anomaly = (actual_label == 0)
            
            norm_features = (features - mu_flat) / sigma_flat
            
            if len(history_buffer) == CFG["window_size"]:
                window_arr = np.array(history_buffer) 
                x = torch.tensor(window_arr.T, dtype=torch.float32).unsqueeze(0).to(device)
                
                pred = model(x).squeeze(0).cpu().numpy()
                score = np.abs(pred - norm_features).mean()
                
                is_anomaly = bool(score > threshold)
                total_predictions += 1
                
                # --- Metrics Logic ---
                if is_anomaly == truth_is_anomaly:
                    correct_predictions += 1
                    
                if is_anomaly and truth_is_anomaly:
                    true_positives += 1
                elif is_anomaly and not truth_is_anomaly:
                    false_positives += 1
                elif not is_anomaly and truth_is_anomaly:
                    false_negatives += 1
                # ---------------------

                if is_anomaly or truth_is_anomaly:
                    time.sleep(0.01)
                    alert = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 Normal"
                    truth = "Anomaly" if truth_is_anomaly else "Normal"
                    print(f"  [TimeStep {i:04d}] Score: {score:.4f} | Thresh: {threshold:.4f} | Pred: {alert:<20s} | Truth: {truth}")
                        
            # Slide the window
            history_buffer.append(norm_features)
            if len(history_buffer) > CFG["window_size"]:
                history_buffer.pop(0)

    # Calculate final accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("  STREAM OFFLINE — FINAL METRICS SUMMARY")
    print("=" * 60)
    print(f"  Total Windows Processed : {total_predictions}")
    print(f"  Overall Accuracy        : {accuracy * 100:.2f}%\n")
    print(f"  Correct Predictions     : {correct_predictions}")
    print(f"  True Positives          : {true_positives}")
    print(f"  False Positives (False Alarms) : {false_positives}")
    print(f"  False Negatives (Missed)       : {false_negatives}")
    print(f"  Precision               : {precision * 100:.2f}%")
    print(f"  Recall                  : {recall * 100:.2f}%")
    print(f"  F1-Score                : {f1 * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()