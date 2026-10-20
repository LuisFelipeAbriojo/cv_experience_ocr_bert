"""Baseline 1 - TF-IDF (word 1-2 grams + char 3-5 grams) + Ridge regression.

Numbers and years are KEPT as tokens by default: a year like "1998" is weak but
real evidence (an early year means a long career) and removing it would make the
baseline artificially weak. `keep_digits=False` replaces every digit by "0" so
the effect can be measured (Question 3). The Ridge alpha is tuned on the
validation set only.

Usage:
    python -m src.baseline_tfidf [--no-digits]
"""
from __future__ import annotations

import argparse
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils import CAP, METRICS, MODELS, clip_years, load_split, save_json

ALPHAS = [0.1, 0.3, 1, 3, 10, 30]


def _lower(s: str) -> str:
    return s.lower()


def _lower_no_digits(s: str) -> str:
    return re.sub(r"\d", "0", s.lower())


def _prep(keep_digits: bool):
    # module-level functions (not lambdas) so the fitted pipeline can be pickled
    return _lower if keep_digits else _lower_no_digits


def make_pipeline(alpha: float, keep_digits: bool = True) -> Pipeline:
    prep = _prep(keep_digits)
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), min_df=2,
                                 max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, max_features=100_000, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("ridge", Ridge(alpha=alpha))])


def train(train_df, val_df, text_col="text", keep_digits=True, verbose=True):
    best = None
    for a in ALPHAS:
        pipe = make_pipeline(a, keep_digits).fit(train_df[text_col], train_df["y"] / CAP)
        mae = np.abs(predict(pipe, val_df[text_col]) - val_df["y"]).mean()
        if verbose:
            print(f"  alpha={a:<5} val MAE={mae:.3f}")
        if best is None or mae < best[0]:
            best = (mae, a)
    final = make_pipeline(best[1], keep_digits).fit(train_df[text_col], train_df["y"] / CAP)
    final.best_alpha_, final.val_mae_ = best[1], best[0]
    return final


def predict(model, texts) -> np.ndarray:
    return clip_years(model.predict(list(texts)) * CAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-digits", action="store_true")
    a = ap.parse_args()
    key = "tfidf" if not a.no_digits else "tfidf_nodigits"
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    t0 = time.perf_counter()
    model = train(tr, va, keep_digits=not a.no_digits)
    train_s = time.perf_counter() - t0
    joblib.dump(model, MODELS / "baselines" / f"{key}.joblib")
    t0 = time.perf_counter()
    pred = predict(model, te["text"])
    infer = (time.perf_counter() - t0) / len(te)
    pd.DataFrame({"id": te.id, "y_true": te.y, "y_pred": pred}).to_csv(
        METRICS / f"preds_{key}.csv", index=False)
    pd.DataFrame({"id": va.id, "y_true": va.y, "y_pred": predict(model, va["text"])}).to_csv(
        METRICS / f"preds_{key}_val.csv", index=False)
    save_json({"train_seconds": train_s, "infer_seconds_per_cv": infer,
               "best_alpha": model.best_alpha_, "val_mae": model.val_mae_},
              METRICS / f"timing_{key}.json")
    print(f"best alpha={model.best_alpha_}; test MAE = {np.abs(pred - te.y).mean():.3f}")


if __name__ == "__main__":
    # run through the package import so the pickled pipeline references
    # src.baseline_tfidf._lower (not __main__._lower) and loads anywhere
    from src.baseline_tfidf import main as _main

    _main()
