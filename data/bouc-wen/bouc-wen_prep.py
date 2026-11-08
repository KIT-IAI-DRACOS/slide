"""Convert Bouc-Wen sinesweep and multisine MATLAB data to CSV."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_series(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    u_key = f"uval_{name}"
    y_key = f"yval_{name}"
    u = loadmat(data_dir / f"{u_key}.mat")[u_key].ravel()
    y = loadmat(data_dir / f"{y_key}.mat")[y_key].ravel()
    if len(u) != len(y):
        raise ValueError(f"{name} input and target lengths differ: {len(u)} != {len(y)}")
    return u, y


def save_series(output_path: Path, u: np.ndarray, y: np.ndarray) -> Path:
    pd.DataFrame({"date": np.arange(len(u)), "u": u, "y": y}).to_csv(output_path, index=False)
    return output_path


def prepare_multisine(data_dir: Path) -> Path:
    u, y = load_series(data_dir, "multisine")
    return save_series(data_dir / "bouc_wen_multisine.csv", u, y)


def prepare_combined(data_dir: Path) -> Path:
    u_sinesweep, y_sinesweep = load_series(data_dir, "sinesweep")
    u_multisine, y_multisine = load_series(data_dir, "multisine")
    return save_series(
        data_dir / "bouc_wen_combined.csv",
        np.concatenate((u_sinesweep, u_multisine)),
        np.concatenate((y_sinesweep, y_multisine)),
    )


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    for output_path in (prepare_multisine(data_dir), prepare_combined(data_dir)):
        print(f"Saved {output_path}")
