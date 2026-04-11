"""
Real sensor data loader for GraGOD.

Loads sensor_data.csv, separates features from the Occupancy label,
and splits into train (normal only) and test (full) sets.

Place sensor_data.csv in:  gragod/data/sensor_data.csv

CSV columns expected:
  date, Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy

Occupancy = 1 → anomaly (room occupied with unusual sensor pattern)
Occupancy = 0 → normal
"""

import numpy as np
import pandas as pd
from pathlib import Path


SENSOR_COLS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
LABEL_COL   = "Occupancy"


def load_sensor_csv(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load sensor_data.csv and produce train/test split.

    Strategy:
      - Use normal samples (Occupancy=0) as training data so the model
        learns what "normal" looks like without any anomaly contamination.
      - Use the full dataset as test data with ground-truth labels.

    Returns
    -------
    train_df    : (n_train, n_sensors)  — normal rows only (no Occupancy col)
    test_df     : (n_test,  n_sensors)  — all rows         (no Occupancy col)
    test_labels : (n_test,)             — 0=normal, 1=anomaly
    """
    df = pd.read_csv(csv_path)

    # drop date col; keep only sensor feature columns
    feature_df = df[SENSOR_COLS].copy()
    labels     = df[LABEL_COL].values.astype(int)

    # train = normal only (Occupancy == 0); preserve temporal order
    normal_mask = labels == 0
    train_df    = feature_df[normal_mask].reset_index(drop=True)

    # test  = all data in temporal order
    test_df     = feature_df.reset_index(drop=True)
    test_labels = labels

    print(f"[real_dataset] Loaded: {csv_path}")
    print(f"  Total rows  : {len(df)}")
    print(f"  Sensors     : {SENSOR_COLS}")
    print(f"  Train (normal only): {len(train_df)} rows")
    print(f"  Test  (all)        : {len(test_df)}  rows")
    print(f"  Anomaly ratio      : {labels.mean():.2%}")

    return train_df, test_df, test_labels
