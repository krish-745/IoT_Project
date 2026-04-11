import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


@torch.no_grad()
def compute_anomaly_scores(model, dataset, batch_size=256, device="cpu", reduce="mean"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_errors = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        errors = torch.abs(model(x) - y)
        all_errors.append(errors.cpu().numpy())
    per_sensor = np.concatenate(all_errors, axis=0)
    scores = per_sensor.mean(axis=1) if reduce == "mean" else per_sensor.max(axis=1)
    return per_sensor, scores


def fit_threshold(train_errors, k=3.0, reduce="mean"):
    scores = train_errors.mean(axis=1) if reduce == "mean" else train_errors.max(axis=1)
    threshold = scores.mean() + k * scores.std()
    print(f"  Threshold  μ={scores.mean():.4f}  σ={scores.std():.4f}  → {threshold:.4f}  (k={k})")
    return float(threshold)


def evaluate(scores, labels, threshold, verbose=True):
    n = len(scores)
    labels_trimmed = labels[-n:]
    preds = (scores > threshold).astype(int)
    precision = precision_score(labels_trimmed, preds, zero_division=0)
    recall    = recall_score(labels_trimmed, preds, zero_division=0)
    f1        = f1_score(labels_trimmed, preds, zero_division=0)
    accuracy  = (preds == labels_trimmed).mean()
    try:
        auc = roc_auc_score(labels_trimmed, scores)
    except ValueError:
        auc = 0.0
    results = {
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": auc, "accuracy": accuracy, "threshold": threshold,
        "n_anomalies_detected": int(preds.sum()),
        "n_anomalies_true":     int(labels_trimmed.sum()),
    }
    if verbose:
        print(f"\n{'═'*45}")
        print(f"  EVALUATION RESULTS")
        print(f"{'─'*45}")
        print(f"  Accuracy   : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision  : {precision:.4f}")
        print(f"  Recall     : {recall:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  ROC-AUC    : {auc:.4f}")
        print(f"  Threshold  : {threshold:.4f}")
        print(f"  Detected   : {results['n_anomalies_detected']} / {results['n_anomalies_true']} true anomalies")
        print(f"{'═'*45}\n")
    return results


def find_best_threshold(train_errors, test_scores, test_labels, k_range=(0.5, 5.0), steps=50):
    best_f1 = -1
    best_k  = 2.5
    for k in np.linspace(*k_range, steps):
        threshold = fit_threshold(train_errors, k=k, reduce="mean")
        n = len(test_scores)
        labels_trimmed = test_labels[-n:]
        preds = (test_scores > threshold).astype(int)
        f1 = f1_score(labels_trimmed, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_k  = k
    print(f"\n  Best k={best_k:.2f}  →  F1={best_f1:.4f}")
    return float(fit_threshold(train_errors, k=best_k, reduce="mean"))
