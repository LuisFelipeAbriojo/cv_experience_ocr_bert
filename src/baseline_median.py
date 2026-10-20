"""Baseline 0 - always predict the median years of the training set.

Usage:
    python -m src.baseline_median
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.utils import METRICS, MODELS, load_split, save_json


def train(train_df: pd.DataFrame, val_df=None, **_) -> float:
    return float(np.median(train_df["y"]))


def predict(model: float, texts) -> np.ndarray:
    return np.full(len(texts), model, dtype=float)


def main():
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    t0 = time.perf_counter()
    model = train(tr)
    train_s = time.perf_counter() - t0
    save_json({"median": model}, MODELS / "baselines" / "median.json")
    t0 = time.perf_counter()
    pred = predict(model, te["text"])
    infer = (time.perf_counter() - t0) / len(te)
    for name, df, p in (("", te, pred), ("_val", va, predict(model, va["text"]))):
        pd.DataFrame({"id": df.id, "y_true": df.y, "y_pred": p}).to_csv(
            METRICS / f"preds_median{name}.csv", index=False)
    save_json({"train_seconds": train_s, "infer_seconds_per_cv": infer, "median": model},
              METRICS / "timing_median.json")
    print(f"median = {model}; test MAE = {np.abs(pred - te.y).mean():.3f}")


if __name__ == "__main__":
    main()
