"""
Synthetic multivariate IoT time-series generator.

Generates N correlated sensor signals (normal behaviour) and injects
point / contextual anomalies into a held-out test segment.

Usage:
    from data.synthetic import generate_dataset
    train_df, test_df, test_labels = generate_dataset()
"""

import numpy as np
import pandas as pd


def generate_dataset(
    n_sensors: int = 10,
    n_train: int = 2000,
    n_test: int = 600,
    anomaly_fraction: float = 0.08,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns
    -------
    train_df   : (n_train, n_sensors)  — normal data only
    test_df    : (n_test,  n_sensors)  — normal + anomalous
    test_labels: (n_test,)             — 0 = normal, 1 = anomaly
    """
    rng = np.random.default_rng(seed)

    # ── base signals: mix of sinusoids with correlated noise ──────────────
    t_train = np.arange(n_train)
    t_test  = np.arange(n_train, n_train + n_test)

    def make_signals(t: np.ndarray) -> np.ndarray:
        """Shape: (len(t), n_sensors)"""
        # latent shared processes
        base1 = np.sin(2 * np.pi * t / 50)
        base2 = np.sin(2 * np.pi * t / 30 + 1.0)
        base3 = np.cos(2 * np.pi * t / 80)

        signals = np.zeros((len(t), n_sensors))
        for i in range(n_sensors):
            # each sensor is a weighted combination of base signals
            w = rng.uniform(0.3, 1.0, size=3)
            signals[:, i] = (
                w[0] * base1
                + w[1] * base2
                + w[2] * base3
                + rng.normal(0, 0.05, size=len(t))
            )
        return signals

    train_data = make_signals(t_train)
    test_data  = make_signals(t_test)

    # ── inject anomalies into test set ────────────────────────────────────
    labels = np.zeros(n_test, dtype=int)
    n_anomaly_windows = int(n_test * anomaly_fraction / 10)  # ~10-step bursts

    for _ in range(n_anomaly_windows):
        start = rng.integers(10, n_test - 20)
        length = rng.integers(5, 15)
        end = min(start + length, n_test)

        # affect 1–3 random sensors
        affected = rng.choice(n_sensors, size=rng.integers(1, 4), replace=False)
        anomaly_type = rng.choice(["spike", "shift", "noise"])

        if anomaly_type == "spike":
            test_data[start:end, affected] += rng.uniform(3, 6) * rng.choice([-1, 1])
        elif anomaly_type == "shift":
            test_data[start:end, affected] += rng.uniform(2, 4)
        else:  # noise burst
            test_data[start:end, affected] += rng.normal(0, 2.0, size=(end - start, len(affected)))

        labels[start:end] = 1

    # ── package as DataFrames ─────────────────────────────────────────────
    cols = [f"sensor_{i:02d}" for i in range(n_sensors)]
    train_df = pd.DataFrame(train_data, columns=cols)
    test_df  = pd.DataFrame(test_data,  columns=cols)

    print(f"[synthetic] train={n_train} steps | test={n_test} steps | "
          f"anomaly_ratio={labels.mean():.2%} | sensors={n_sensors}")

    return train_df, test_df, labels
