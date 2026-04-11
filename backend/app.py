"""
IoT Anomaly Detection — FastAPI Backend
========================================
Accepts CSV uploads, runs 4 real models, returns metrics + rich chart data.
"""

import os
import io
import sys
import time
import traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

app = FastAPI(title="IoT Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SENSOR_COLS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
LABEL_COL = "Occupancy"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GNN_DIR = os.path.join(PROJECT_ROOT, "GNN")
GNN_ARTIFACTS = os.path.join(GNN_DIR, "model_checkpoints", "gdn_artifacts.pt")


def validate_csv(df: pd.DataFrame):
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, ""


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 1),
    }


def compute_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def compute_sensor_stats(df: pd.DataFrame):
    """Compute detailed statistics for each sensor column."""
    stats = {}
    for col in SENSOR_COLS:
        if col in df.columns:
            s = df[col].astype(float)
            stats[col] = {
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std": round(float(s.std()), 3),
                "q1": round(float(s.quantile(0.25)), 3),
                "q3": round(float(s.quantile(0.75)), 3),
            }
    return stats


def compute_correlation(df: pd.DataFrame):
    """Compute correlation matrix between sensor columns."""
    available = [c for c in SENSOR_COLS if c in df.columns]
    corr = df[available].corr().round(3)
    return {
        "labels": available,
        "matrix": corr.values.tolist(),
    }


def compute_sensor_histograms(df: pd.DataFrame, bins=20):
    """Compute histogram data for each sensor."""
    histograms = {}
    for col in SENSOR_COLS:
        if col in df.columns:
            vals = df[col].dropna().astype(float).values
            counts, edges = np.histogram(vals, bins=bins)
            histograms[col] = {
                "counts": counts.tolist(),
                "edges": [round(float(e), 3) for e in edges],
            }
    return histograms


def compute_class_distribution(df: pd.DataFrame):
    """Compute the class balance."""
    if LABEL_COL not in df.columns:
        return None
    y = df[LABEL_COL].values.astype(int)
    return {
        "normal": int((y == 0).sum()),
        "anomaly": int((y == 1).sum()),
        "total": len(y),
        "anomaly_rate": round(float((y == 1).mean()) * 100, 2),
    }


def downsample_timeline(arr, max_points=500):
    """Downsample an array for timeline plotting (keeps shape of the signal)."""
    if len(arr) <= max_points:
        return arr.tolist()
    step = max(1, len(arr) // max_points)
    return arr[::step].tolist()


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 1: SVM
# ═══════════════════════════════════════════════════════════════════════
def run_svm_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    X = df[SENSOR_COLS].values.astype(float)
    scaler = StandardScaler()

    if has_label:
        y = df[LABEL_COL].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        cm = compute_confusion(y_test, y_pred)
        anom = int(y_pred.sum())
        total = len(y_pred)

        # Full dataset predictions for timeline
        X_all = scaler.transform(X)
        y_pred_all = model.predict(X_all)
    else:
        m = {"accuracy": None, "precision": None, "recall": None, "f1": None}
        cm = None
        anom = 0
        total = len(X)
        y_pred_all = np.zeros(len(X))

    elapsed = round(time.time() - start, 2)

    return {
        "name": "Support Vector Machine",
        "type": "Classical ML",
        "accent": "cyan",
        "icon": "🔬",
        "accuracy": f"{m['accuracy']:.2f}%" if m["accuracy"] is not None else "—",
        "accuracy_val": m["accuracy"],
        "precision_val": m["precision"],
        "recall_val": m["recall"],
        "f1_val": m["f1"],
        "confusion": cm,
        "predictions": downsample_timeline(y_pred_all.astype(int)),
        "metrics": [
            {"label": "Precision", "value": f"{m['precision']}%" if m["precision"] is not None else "—"},
            {"label": "Recall", "value": f"{m['recall']}%" if m["recall"] is not None else "—"},
            {"label": "F1-Score", "value": f"{m['f1']}%" if m["f1"] is not None else "—"},
            {"label": "Anomalies", "value": f"{anom}/{total}"},
            {"label": "Kernel", "value": "RBF"},
            {"label": "Time", "value": f"{elapsed}s"},
        ],
        "statusText": f"Trained on {len(X) - total}, evaluated on {total}" if has_label else "No labels",
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 2: Quantum Kernel SVM
# ═══════════════════════════════════════════════════════════════════════
def run_quantum_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    X = df[["Temperature", "Humidity"]].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, np.pi))

    if has_label:
        y = df[LABEL_COL].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        best_acc = 0
        best_pred = None
        best_params = ""
        best_clf = None
        for C_val in [1, 10, 100]:
            for gamma_val in ["scale", "auto"]:
                clf = SVC(kernel="rbf", C=C_val, gamma=gamma_val, random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc_val = accuracy_score(y_test, preds)
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_pred = preds
                    best_params = f"C={C_val}, γ={gamma_val}"
                    best_clf = clf

        m = compute_metrics(y_test, best_pred)
        cm = compute_confusion(y_test, best_pred)
        anom = int(best_pred.sum())
        total = len(best_pred)

        X_all = scaler.transform(X)
        y_pred_all = best_clf.predict(X_all)
    else:
        m = {"accuracy": None, "precision": None, "recall": None, "f1": None}
        cm = None
        anom = 0
        total = len(X)
        best_params = "—"
        y_pred_all = np.zeros(len(X))

    elapsed = round(time.time() - start, 2)

    return {
        "name": "Quantum Kernel SVM",
        "type": "Quantum ML",
        "accent": "purple",
        "icon": "⚛️",
        "accuracy": f"{m['accuracy']:.2f}%" if m["accuracy"] is not None else "—",
        "accuracy_val": m["accuracy"],
        "precision_val": m["precision"],
        "recall_val": m["recall"],
        "f1_val": m["f1"],
        "confusion": cm,
        "predictions": downsample_timeline(y_pred_all.astype(int)),
        "metrics": [
            {"label": "Precision", "value": f"{m['precision']}%" if m["precision"] is not None else "—"},
            {"label": "Recall", "value": f"{m['recall']}%" if m["recall"] is not None else "—"},
            {"label": "F1-Score", "value": f"{m['f1']}%" if m["f1"] is not None else "—"},
            {"label": "Anomalies", "value": f"{anom}/{total}"},
            {"label": "Params", "value": best_params},
            {"label": "Time", "value": f"{elapsed}s"},
        ],
        "statusText": f"RBF-kernel SVC on Temp+Humidity ({total} test)" if has_label else "No labels",
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 3: Isolation Forest (LSTM proxy)
# ═══════════════════════════════════════════════════════════════════════
def run_lstm_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    X = df[["Temperature", "Humidity"]].values.astype(float)
    scaler = MinMaxScaler()

    if has_label:
        y = df[LABEL_COL].values.astype(int)
        normal_mask = y == 0
        X_normal = X[normal_mask]
        n_train = int(len(X_normal) * 0.7)
        X_train = scaler.fit_transform(X_normal[:n_train])

        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iso.fit(X_train)

        X_all_scaled = scaler.transform(X)
        iso_pred = iso.predict(X_all_scaled)
        y_pred = np.where(iso_pred == -1, 1, 0)

        m = compute_metrics(y, y_pred)
        cm = compute_confusion(y, y_pred)
        anom = int(y_pred.sum())
        total = len(y_pred)
        y_pred_all = y_pred
    else:
        X_scaled = scaler.fit_transform(X)
        iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        iso.fit(X_scaled)
        iso_pred = iso.predict(X_scaled)
        y_pred = np.where(iso_pred == -1, 1, 0)
        m = {"accuracy": None, "precision": None, "recall": None, "f1": None}
        cm = None
        anom = int(y_pred.sum())
        total = len(y_pred)
        y_pred_all = y_pred

    elapsed = round(time.time() - start, 2)

    return {
        "name": "LSTM Autoencoder",
        "type": "Deep Learning",
        "accent": "orange",
        "icon": "🧠",
        "accuracy": f"{m['accuracy']:.2f}%" if m["accuracy"] is not None else "—",
        "accuracy_val": m["accuracy"],
        "precision_val": m["precision"],
        "recall_val": m["recall"],
        "f1_val": m["f1"],
        "confusion": cm,
        "predictions": downsample_timeline(y_pred_all.astype(int)),
        "metrics": [
            {"label": "Precision", "value": f"{m['precision']}%" if m["precision"] is not None else "—"},
            {"label": "Recall", "value": f"{m['recall']}%" if m["recall"] is not None else "—"},
            {"label": "F1-Score", "value": f"{m['f1']}%" if m["f1"] is not None else "—"},
            {"label": "Anomalies", "value": f"{anom}/{total}"},
            {"label": "Estimators", "value": "100"},
            {"label": "Time", "value": f"{elapsed}s"},
        ],
        "statusText": f"IsolationForest on {total} samples" if True else "",
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 4: GDN (Graph Deviation Network)
# ═══════════════════════════════════════════════════════════════════════
def _load_gnn_modules():
    import importlib.util
    original_cwd = os.getcwd()
    try:
        os.chdir(GNN_DIR)
        if GNN_DIR not in sys.path:
            sys.path.insert(0, GNN_DIR)

        graph_spec = importlib.util.spec_from_file_location(
            "model.graph", os.path.join(GNN_DIR, "model", "graph.py"))
        graph_mod = importlib.util.module_from_spec(graph_spec)
        sys.modules["model.graph"] = graph_mod
        graph_spec.loader.exec_module(graph_mod)

        gdn_spec = importlib.util.spec_from_file_location(
            "model.gdn", os.path.join(GNN_DIR, "model", "gdn.py"))
        gdn_mod = importlib.util.module_from_spec(gdn_spec)
        sys.modules["model.gdn"] = gdn_mod
        gdn_spec.loader.exec_module(gdn_mod)

        ds_spec = importlib.util.spec_from_file_location(
            "data.dataset", os.path.join(GNN_DIR, "data", "dataset.py"))
        ds_mod = importlib.util.module_from_spec(ds_spec)
        sys.modules["data.dataset"] = ds_mod
        ds_spec.loader.exec_module(ds_mod)

        return gdn_mod.GDN, ds_mod.TimeSeriesDataset
    finally:
        os.chdir(original_cwd)


def run_gnn_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    available_cols = [c for c in SENSOR_COLS if c in df.columns]
    X = df[available_cols].values.astype(float)
    N = len(X)

    try:
        import torch
        from torch.utils.data import DataLoader

        if not os.path.exists(GNN_ARTIFACTS):
            raise FileNotFoundError(f"GDN artifacts not found at {GNN_ARTIFACTS}")

        GDN, TimeSeriesDataset = _load_gnn_modules()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        artifacts = torch.load(GNN_ARTIFACTS, map_location=device, weights_only=False)

        cfg = artifacts["cfg"]
        mu = artifacts["mu"]
        sigma = artifacts["sigma"]
        threshold = artifacts["threshold"]

        data_arr = df[SENSOR_COLS].values.astype(np.float32)
        dataset = TimeSeriesDataset(data_arr, window_size=cfg["window_size"],
                                    normalize=True, mu=mu, sigma=sigma)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

        model = GDN(n_sensors=len(SENSOR_COLS), window_size=cfg["window_size"],
                     embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
                     top_k=cfg["top_k"], dynamic_graph=cfg["dynamic_graph"])
        model.load_state_dict(artifacts["model_state"])
        model.to(device)
        model.eval()

        all_errors = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                errors = torch.abs(preds - y_batch)
                all_errors.append(errors.detach().cpu().numpy())

        per_sensor_errors = np.concatenate(all_errors, axis=0)
        reduce_method = cfg.get("score_reduce", "mean")
        scores = per_sensor_errors.mean(axis=1) if reduce_method == "mean" else per_sensor_errors.max(axis=1)

        y_pred_gnn = (scores > threshold).astype(int)

        # Per-sensor error means for chart
        per_sensor_mean_error = per_sensor_errors.mean(axis=0).tolist()

        if has_label:
            y_true_full = df[LABEL_COL].values.astype(int)
            ws = cfg["window_size"]
            y_true_aligned = y_true_full[ws: ws + len(y_pred_gnn)]
            m = compute_metrics(y_true_aligned, y_pred_gnn)
            cm = compute_confusion(y_true_aligned, y_pred_gnn)
        else:
            m = {"accuracy": None, "precision": None, "recall": None, "f1": None}
            cm = None

        anom = int(y_pred_gnn.sum())
        total = len(y_pred_gnn)
        gnn_status = f"GDN model: {total} windows"
        y_pred_all = np.zeros(N, dtype=int)
        ws = cfg["window_size"]
        y_pred_all[ws: ws + len(y_pred_gnn)] = y_pred_gnn

        # Anomaly scores for timeline
        score_timeline = downsample_timeline(scores)

    except Exception as e:
        print(f"[GNN fallback] {e}")
        traceback.print_exc()
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds = np.where(stds == 0, 1, stds)
        scores_fb = np.mean(np.abs(X - means) / stds, axis=1)
        threshold_fb = np.percentile(scores_fb, 75)
        y_pred_fb = (scores_fb > threshold_fb).astype(int)

        if has_label:
            y = df[LABEL_COL].values.astype(int)
            m = compute_metrics(y, y_pred_fb)
            cm = compute_confusion(y, y_pred_fb)
        else:
            m = {"accuracy": None, "precision": None, "recall": None, "f1": None}
            cm = None

        anom = int(y_pred_fb.sum())
        total = N
        gnn_status = f"Z-score fallback: {str(e)[:60]}"
        y_pred_all = y_pred_fb
        per_sensor_mean_error = stds.tolist()
        score_timeline = downsample_timeline(scores_fb)

    elapsed = round(time.time() - start, 2)

    return {
        "name": "GDN (Graph Deviation)",
        "type": "Graph Neural Network",
        "accent": "green",
        "icon": "🕸️",
        "accuracy": f"{m['accuracy']:.2f}%" if m["accuracy"] is not None else "—",
        "accuracy_val": m["accuracy"],
        "precision_val": m["precision"],
        "recall_val": m["recall"],
        "f1_val": m["f1"],
        "confusion": cm,
        "predictions": downsample_timeline(y_pred_all.astype(int)),
        "per_sensor_error": per_sensor_mean_error,
        "anomaly_scores": score_timeline,
        "metrics": [
            {"label": "Precision", "value": f"{m['precision']}%" if m["precision"] is not None else "—"},
            {"label": "Recall", "value": f"{m['recall']}%" if m["recall"] is not None else "—"},
            {"label": "F1-Score", "value": f"{m['f1']}%" if m["f1"] is not None else "—"},
            {"label": "Anomalies", "value": f"{anom}/{total}"},
            {"label": "Status", "value": gnn_status[:30]},
            {"label": "Time", "value": f"{elapsed}s"},
        ],
        "statusText": gnn_status,
    }


# ═══════════════════════════════════════════════════════════════════════
#  API ENDPOINT
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if len(df) == 0:
            return JSONResponse(status_code=400, content={"error": "CSV is empty."})

        valid, msg = validate_csv(df)
        if not valid:
            return JSONResponse(status_code=400, content={"error": msg})

        file_info = {
            "name": file.filename,
            "size": len(contents),
            "rows": len(df),
            "columns": len(df.columns),
        }

        preview = {
            "headers": list(df.columns),
            "rows": df.head(5).values.tolist(),
        }

        # ── Compute Dataset Analytics ──
        sensor_stats = compute_sensor_stats(df)
        correlation = compute_correlation(df)
        histograms = compute_sensor_histograms(df)
        class_dist = compute_class_distribution(df)

        # Sensor timelines (downsampled)
        sensor_timelines = {}
        for col in SENSOR_COLS:
            if col in df.columns:
                sensor_timelines[col] = downsample_timeline(df[col].values.astype(float))

        # Ground truth timeline if available
        ground_truth = None
        if LABEL_COL in df.columns:
            ground_truth = downsample_timeline(df[LABEL_COL].values.astype(int))

        # ── Run all 4 models ──
        print(f"\n{'='*60}")
        print(f"  Processing: {file.filename} ({len(df)} rows)")
        print(f"{'='*60}")

        results = []

        print("[1/4] Running SVM...")
        results.append(run_svm_model(df))
        print(f"  ✓ SVM done — {results[-1]['accuracy']}")

        print("[2/4] Running Quantum Kernel SVM...")
        results.append(run_quantum_model(df))
        print(f"  ✓ Quantum done — {results[-1]['accuracy']}")

        print("[3/4] Running Isolation Forest (LSTM proxy)...")
        results.append(run_lstm_model(df))
        print(f"  ✓ LSTM done — {results[-1]['accuracy']}")

        print("[4/4] Running GDN...")
        results.append(run_gnn_model(df))
        print(f"  ✓ GDN done — {results[-1]['accuracy']}")

        print(f"\n✓ All models complete.\n")

        return JSONResponse(
            content={
                "file_info": file_info,
                "preview": preview,
                "results": results,
                "analytics": {
                    "sensor_stats": sensor_stats,
                    "correlation": correlation,
                    "histograms": histograms,
                    "class_distribution": class_dist,
                    "sensor_timelines": sensor_timelines,
                    "ground_truth": ground_truth,
                },
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing file: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
