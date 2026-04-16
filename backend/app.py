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
from sklearn.svm import SVC, OneClassSVM
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

# UPDATED: Restricted to only 2 features
SENSOR_COLS = ["Temperature", "Humidity"]
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
        if isinstance(arr, list):
            return arr
        return arr.tolist()
    step = max(1, len(arr) // max_points)
    return arr[::step].tolist() if not isinstance(arr, list) else arr[::step]


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 1: SVM
# ═══════════════════════════════════════════════════════════════════════
def run_svm_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    X = df[SENSOR_COLS].values.astype(float)
    scaler = MinMaxScaler()

    if has_label:
        y = df[LABEL_COL].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        cm = compute_confusion(y_test, y_pred)
        anom = int(y_pred.sum())
        total = len(y_pred)

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
#  MODEL 2: Quantum Kernel SVM (Updated to match IoT_Quantum.ipynb)
# ═══════════════════════════════════════════════════════════════════════
def run_quantum_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    X = df[SENSOR_COLS].values.astype(float)
    
    try:
        import pennylane as qml
        import pennylane.numpy as pnp
    except ImportError:
        print("PennyLane not found. Please `pip install pennylane`.")
        return {
            "name": "Quantum Kernel SVM",
            "type": "Quantum ML",
            "accent": "purple",
            "icon": "⚛️",
            "accuracy": "—",
            "accuracy_val": None,
            "precision_val": None,
            "recall_val": None,
            "f1_val": None,
            "confusion": None,
            "predictions": [],
            "metrics": [],
            "statusText": "PennyLane library missing.",
        }

    # ── Kept small for speed (matched from notebook) ────────────────────────
    SEED = 42
    N_QUBITS    = 2
    N_LAYERS    = 2
    N_STEPS     = 30
    BATCH_SIZE  = 15
    LR          = 0.05
    N_TRAIN     = 30
    N_TEST_OK   = 15
    N_TEST_ANOM = 10

    np.random.seed(SEED)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    if has_label:
        # Notebook logic: Normal is +1, Anomaly is -1
        y = np.where(df[LABEL_COL] == 0, 1, -1)

        nidx = np.where(y == 1)[0]
        aidx = np.where(y == -1)[0]

        rng = np.random.default_rng(SEED)
        rng.shuffle(nidx)
        rng.shuffle(aidx)

        # Safely enforce notebook's split rules relative to dataset size
        actual_n_train = min(N_TRAIN, len(nidx))
        X_train = X_scaled[nidx[:actual_n_train]]
        y_train = y[nidx[:actual_n_train]]

        rem_nidx = nidx[actual_n_train:]
        actual_test_ok = min(N_TEST_OK, len(rem_nidx))
        actual_test_anom = min(N_TEST_ANOM, len(aidx))

        test_idx = np.concatenate([rem_nidx[:actual_test_ok], aidx[:actual_test_anom]])
        X_test   = X_scaled[test_idx]
        y_test_q = y[test_idx]

        perm   = rng.permutation(len(y_test_q))
        X_test = X_test[perm]
        y_test_q = y_test_q[perm]

        dev = qml.device("default.qubit", wires=N_QUBITS)

        def ansatz(x, params):
            """Simple 2-qubit ansatz: data encoding + trainable rotations + CNOT."""
            for layer in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.RX(x[q % len(x)], wires=q)
                    qml.RY(params[layer, q, 0], wires=q)
                    qml.RZ(params[layer, q, 1], wires=q)
                qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def kernel_circuit(x1, x2, params):
            ansatz(x1, params)
            qml.adjoint(ansatz)(x2, params)
            return qml.probs(wires=range(N_QUBITS))

        def kernel_fn(x1, x2, params):
            # No float() cast — keeps autograd tape intact
            return kernel_circuit(x1, x2, params)[0]

        def kta_loss(params, X_batch, y_batch):
            K = qml.kernels.square_kernel_matrix(
                X_batch,
                lambda x1, x2: kernel_fn(x1, x2, params),
                assume_normalized_kernel=False
            )
            # Correct ideal kernel: T_ij = y_i * y_j  (labels must be ±1)
            y_b = y_batch.reshape(-1, 1).astype(float)
            T = y_b @ y_b.T
            # Frobenius inner product <K, T>
            kta = pnp.sum(K * T) / (pnp.linalg.norm(K) * pnp.linalg.norm(T) + 1e-8)
            return -kta

        params = pnp.array(
            np.random.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS, 2)),
            requires_grad=True
        )
        opt = qml.AdamOptimizer(stepsize=LR)

        # ── KTA training ───────────────────────────────────────────────────
        for step in range(N_STEPS):
            curr_batch = min(BATCH_SIZE, len(X_train))
            idx = np.random.choice(len(X_train), curr_batch, replace=False)
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            def cost(p):
                return kta_loss(p, X_batch, y_batch)

            params, loss_val = opt.step_and_cost(cost, params)

        # ── Build kernel matrices & classify ────────────────────────────────
        K_train = qml.kernels.square_kernel_matrix(
            X_train, lambda x1, x2: kernel_fn(x1, x2, params),
            assume_normalized_kernel=True
        )
        K_test = qml.kernels.kernel_matrix(
            X_test, X_train, lambda x1, x2: kernel_fn(x1, x2, params)
        )

        # Auto-scale
        scale   = 1.0 / (K_train.mean() + 1e-8)
        K_train_s = K_train * scale
        K_test_s  = K_test  * scale

        # Grid-search nu
        best_acc, best_pred = 0, None
        best_nu = 0.05
        best_clf = None
        for nu in [0.05, 0.10, 0.15, 0.20]:
            clf = OneClassSVM(kernel="precomputed", nu=nu)
            preds = clf.fit(K_train_s).predict(K_test_s)
            acc   = accuracy_score(y_test_q, preds)
            
            if acc > best_acc or best_pred is None:
                best_acc, best_pred = acc, preds
                best_nu = nu
                best_clf = clf

        # Convert outputs back to standard app format (0 = normal, 1 = anomaly)
        y_pred_app = np.where(best_pred == 1, 0, 1)
        y_test_app = np.where(y_test_q == 1, 0, 1)

        m = compute_metrics(y_test_app, y_pred_app)
        cm = compute_confusion(y_test_app, y_pred_app)
        anom = int(y_pred_app.sum())
        total = len(y_pred_app)

        # Compute full-timeline predictions
        K_all = qml.kernels.kernel_matrix(
            X_scaled, X_train, lambda x1, x2: kernel_fn(x1, x2, params)
        )
        K_all_s = K_all * scale
        y_pred_all_q = best_clf.predict(K_all_s)
        y_pred_all = np.where(y_pred_all_q == 1, 0, 1)
        best_params = f"KTA Train (ν={best_nu})"

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
        "statusText": f"Pennylane Q-Kernel Trained ({total} test)" if has_label else "No labels",
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODEL 3: Deep LSTM Autoencoder (TensorFlow/Keras)
# ═══════════════════════════════════════════════════════════════════════
def run_lstm_model(df: pd.DataFrame):
    start = time.time()
    has_label = LABEL_COL in df.columns
    merged_data = df[SENSOR_COLS + ([LABEL_COL] if has_label else [])].values.astype(float)

    try:
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, RepeatVector, TimeDistributed, LayerNormalization
        )
        from tensorflow.keras.models import Model
        from sklearn.metrics import f1_score
        tf.get_logger().setLevel('ERROR')
        from numpy.random import seed
        seed(10)
        tf.random.set_seed(10)
        scaler = MinMaxScaler()
        WINDOW = 5

        def create_sequences(data, window):
            X = []
            for i in range(len(data) - window):
                X.append(data[i : i + window])
            return np.array(X)

        # Retaining script's hardcoded splits (149 & 162) if the CSV is large enough
        train_end = 149 if len(merged_data) > 162 else max(int(len(merged_data)*0.6), WINDOW+1)
        val_end = 162 if len(merged_data) > 162 else min(train_end + WINDOW + 5, len(merged_data) - WINDOW - 1)

        train = pd.DataFrame(merged_data[:train_end, :2], columns=['Temperature', 'Humidity'])

        if has_label:
            test_full = pd.DataFrame(merged_data[val_end:, :], columns=['Temperature', 'Humidity', 'Occupancy'])
            test = test_full[['Temperature', 'Humidity']].copy()
            y_test = test_full['Occupancy'].astype(int).values
            
            val_full = pd.DataFrame(merged_data[train_end:val_end,:],columns=['Temperature', 'Humidity', 'Occupancy'])
            val = val_full[['Temperature', 'Humidity']].copy()
            y_val = val_full['Occupancy'].astype(int).values
        else:
            test = pd.DataFrame(merged_data[val_end:, :2], columns=['Temperature', 'Humidity'])
            val = pd.DataFrame(merged_data[train_end:val_end,:2], columns=['Temperature', 'Humidity'])
            y_test = np.zeros(len(test), dtype=int)
            y_val = np.zeros(len(val), dtype=int)

        X_train = create_sequences(scaler.fit_transform(train.values), WINDOW)
        X_test = create_sequences(scaler.transform(test.values), WINDOW)
        X_val = create_sequences(scaler.transform(val.values), WINDOW)

        def autoencoder_model(X):
            inputs = Input(shape=(X.shape[1], X.shape[2]))
            L1 = LSTM(32, return_sequences=True)(inputs)
            L1 = LayerNormalization()(L1)
            L2 = LSTM(16, return_sequences=True)(L1)
            L2 = LayerNormalization()(L2)
            L3 = LSTM(4, return_sequences=False)(L2)
            L4 = RepeatVector(X.shape[1])(L3)
            L5 = LSTM(16, return_sequences=True)(L4)
            L5 = LayerNormalization()(L5)
            L6 = LSTM(32, return_sequences=True)(L5)
            output = TimeDistributed(Dense(X.shape[2]))(L6)

            model = Model(inputs=inputs, outputs=output)
            return model

        model = autoencoder_model(X_train)
        model.compile(optimizer='adam', loss='mae')

        nb_epochs = 50
        batch_size = 10
        history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                            validation_split=0.05, verbose=0).history

        X_pred = model.predict(X_test, verbose=0)
        error_vectors = np.abs(X_pred - X_test)
        error_vectors = error_vectors.reshape(error_vectors.shape[0], -1)

        X_pred_train = model.predict(X_train, verbose=0)
        error_train = np.abs(X_pred_train - X_train)
        error_train = error_train.reshape(error_train.shape[0], -1)
        
        mu = np.mean(error_train, axis=0)
        sigma = np.cov(error_train, rowvar=False)
        sigma += 1e-6 * np.eye(sigma.shape[0])

        inv_sigma = np.linalg.inv(sigma)
        
        scores = []
        for e in error_vectors:
            diff = e - mu
            score = diff.T @ inv_sigma @ diff
            scores.append(score)
        scores = np.array(scores)

        X_val_pred = model.predict(X_val, verbose=0)
        error_val = np.abs(X_val_pred - X_val)
        error_val = error_val.reshape(error_val.shape[0], -1)

        val_scores = []
        for e in error_val:
            diff = e - mu
            score = diff.T @ inv_sigma @ diff
            val_scores.append(score)
        val_scores = np.array(val_scores)

        window_size = WINDOW
        y_val_window = []
        for i in range(len(y_val) - window_size ):
            window = y_val[i:i+window_size]
            label = 1 if np.any(window == 1) else 0
            y_val_window.append(label)
        y_val_window = np.array(y_val_window)

        best_f1 = 0
        best_threshold = 0
        for t in np.linspace(min(val_scores), max(val_scores), 100):
            y_pred_val_temp = (val_scores > t).astype(int)
            f1 = f1_score(y_val_window, y_pred_val_temp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        threshold = best_threshold

        y_test_window = []
        for i in range(len(y_test) - window_size ):
            window = y_test[i:i+window_size]
            label = 1 if np.any(window == 1) else 0
            y_test_window.append(label)
        y_test_window = np.array(y_test_window)

        scored = pd.DataFrame()
        scored['AnomalyScore'] = scores
        scored['Threshold'] = threshold
        scored['Anomaly'] = scored['AnomalyScore'] > threshold
        y_test_aligned = y_test_window
        scored['Actual Anomaly'] = y_test_aligned.astype(bool)

        y_pred = scored['Anomaly'].astype(int).values

        if has_label:
            y_true = y_test_aligned
            m = compute_metrics(y_true, y_pred)
            cm = compute_confusion(y_true, y_pred)
        else:
            m = {"accuracy": None, "precision": None, "recall": None, "f1": None}
            cm = None

        anom = int(y_pred.sum())
        total = len(y_pred)

        # Aligning the windowed predictions back to original timeline length
        y_pred_all = np.zeros(len(merged_data), dtype=int)
        offset = val_end + WINDOW
        y_pred_all[offset: offset + len(y_pred)] = y_pred

        status_txt = f"LSTM-AE · threshold={threshold:.1f} (val F1={best_f1*100:.1f}%)"

    except ImportError:
        print("TensorFlow not found. Using IsolationForest fallback.")
        scaler = MinMaxScaler()
        X_feat = df[SENSOR_COLS].values.astype(float)
        X_scaled = scaler.fit_transform(X_feat)
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        iso.fit(X_scaled)
        y_pred_all = np.where(iso.predict(X_scaled) == -1, 1, 0)

        if has_label:
            y  = df[LABEL_COL].values.astype(int)
            m  = compute_metrics(y, y_pred_all)
            cm = compute_confusion(y, y_pred_all)
        else:
            m  = {"accuracy": None, "precision": None, "recall": None, "f1": None}
            cm = None

        anom       = int(y_pred_all.sum())
        total      = len(y_pred_all)
        status_txt = "Fallback: IsolationForest"

    elapsed = round(time.time() - start, 2)

    return {
        "name": "LSTM Autoencoder",
        "type": "Deep Learning",
        "accent": "orange",
        "icon": "🧠",
        "accuracy": f"{m['accuracy']:.2f}%" if m['accuracy'] is not None else "—",
        "accuracy_val": m["accuracy"],
        "precision_val": m["precision"],
        "recall_val": m["recall"],
        "f1_val": m["f1"],
        "confusion": cm,
        "predictions": downsample_timeline(y_pred_all.astype(int)),
        "metrics": [
            {"label": "Precision",    "value": f"{m['precision']}%" if m["precision"] is not None else "—"},
            {"label": "Recall",       "value": f"{m['recall']}%"    if m["recall"]    is not None else "—"},
            {"label": "F1-Score",     "value": f"{m['f1']}%"        if m["f1"]        is not None else "—"},
            {"label": "Anomalies",    "value": f"{anom}/{total}"},
            {"label": "Architecture", "value": "LSTM 32→16→4 + Mahalanobis"},
            {"label": "Time",         "value": f"{elapsed}s"},
        ],
        "statusText": status_txt,
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

        # Automatically adapts n_sensors to len(SENSOR_COLS) which is now 2
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

        print("[3/4] Running Deep LSTM Autoencoder...")
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