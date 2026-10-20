"""Optional bonus - rule-based model: extract job date ranges and add them up.

Every "YYYY.MM - YYYY.MM" (or "- present") range in the work-history part of the
CV is turned into an interval in months; overlapping intervals are merged so
parallel jobs are not double-counted; the total is converted to years. No
training at all, so it is an honest test of whether BERT does better than
simple arithmetic. Education years are ignored (a single year, not a range).

Usage:
    python -m src.baseline_rules
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from src.utils import METRICS, clip_years, load_split, save_json

PRESENT = (2023, 1)  # latest end date that appears in PopResume CVs
RANGE = re.compile(r"((?:19|20)\d\d)\s*[.,]\s*(\d{1,2})\s*[-–—~]+\s*"
                   r"(?:((?:19|20)\d\d)\s*[.,]\s*(\d{1,2})|(present|current|now))", re.I)
EDU = re.compile(r"(?im)^\s*education\s*$")


def intervals(text: str):
    m = EDU.search(text)
    work = text[:m.start()] if m else text
    out = []
    for y1, m1, y2, m2, now in RANGE.findall(work):
        start = int(y1) * 12 + min(max(int(m1), 1), 12) - 1
        end = (PRESENT[0] * 12 + PRESENT[1] - 1) if now else int(y2) * 12 + min(max(int(m2), 1), 12) - 1
        if end >= start:
            out.append((start, end))
    return out


def total_years(text: str) -> float:
    ivs = sorted(intervals(text))
    if not ivs:
        return np.nan
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged) / 12


def train(train_df, val_df=None, text_col="text", **_):
    # the only "fit": value used when no date range was found (median of train)
    return float(np.median(train_df["y"]))


def predict(model, texts) -> np.ndarray:
    preds = np.array([total_years(t) for t in texts], dtype=float)
    return clip_years(np.where(np.isnan(preds), model, preds))


def main():
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    fallback = train(tr)
    t0 = time.perf_counter()
    pred = predict(fallback, te["text"])
    infer = (time.perf_counter() - t0) / len(te)
    no_dates = int(np.isnan([total_years(t) for t in te["text"]]).sum())
    pd.DataFrame({"id": te.id, "y_true": te.y, "y_pred": pred}).to_csv(
        METRICS / "preds_rules.csv", index=False)
    pd.DataFrame({"id": va.id, "y_true": va.y, "y_pred": predict(fallback, va["text"])}).to_csv(
        METRICS / "preds_rules_val.csv", index=False)
    save_json({"train_seconds": 0.0, "infer_seconds_per_cv": infer,
               "test_cvs_without_dates": no_dates}, METRICS / "timing_rules.json")
    print(f"test MAE = {np.abs(pred - te.y).mean():.3f}; CVs with no dates found: {no_dates}")


if __name__ == "__main__":
    main()
