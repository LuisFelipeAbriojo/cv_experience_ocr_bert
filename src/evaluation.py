"""Stage 6 - Metrics, paired bootstrap and the final model comparison.

Every model script writes its test predictions to
output/metrics/preds_<model>.csv (columns id, y_true, y_pred) and its timings to
output/metrics/timing_<model>.json. This module turns them into:
  * output/tables/model_comparison.csv
  * output/tables/bert_gain_bootstrap.csv     (paired bootstrap, 95% CI)
  * output/tables/mae_by_band.csv, mae_by_occupation.csv
  * output/tables/confusion_<model>.csv
  * output/tables/fairness_bert.csv
  * output/tables/bert_top10_errors.csv

Usage:
    python -m src.evaluation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score

from src.utils import BAND_LABELS, METRICS, RAW, SEED, TABLES, clip_years, load_json, to_band

MODELS = {
    "median": "Median",
    "tfidf": "TF-IDF + Ridge",
    "embeddings": "Embeddings + Ridge",
    "rules": "Date rules (bonus)",
    "bert": "Fine-tuned BERT",
}


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = clip_years(y_pred)
    err = np.abs(y_pred - y_true)
    bt, bp = to_band(y_true), to_band(y_pred)
    return {
        "MAE (years)": err.mean(),
        "RMSE (years)": np.sqrt(((y_pred - y_true) ** 2).mean()),
        "Acc ±1": (err <= 1).mean(),
        "Acc ±2": (err <= 2).mean(),
        "Acc ±3": (err <= 3).mean(),
        "Band accuracy": accuracy_score(bt, bp),
        "Band macro-F1": f1_score(bt, bp, labels=BAND_LABELS, average="macro", zero_division=0),
        "R²": r2_score(y_true, y_pred),
        "Mean error (pred-true)": (y_pred - y_true).mean(),
    }


def paired_bootstrap(y, pred_a, pred_b, n=2000, seed=SEED):
    """MAE(b) - MAE(a) on resampled test sets. Positive = model a is better."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(clip_years(pred_a) - y), np.abs(clip_years(pred_b) - y)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = eb[idx].mean(1) - ea[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(gain=eb.mean() - ea.mean(), ci_low=lo, ci_high=hi,
                p_value=float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
                significant=bool(lo > 0 or hi < 0))


def load_preds(split: str = "test", suffix: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for key in MODELS:
        p = METRICS / f"preds_{key}{suffix}{'' if split == 'test' else '_' + split}.csv"
        if p.exists():
            out[key] = pd.read_csv(p, dtype={"id": str}).sort_values("id").reset_index(drop=True)
    return out


def comparison_table(preds) -> pd.DataFrame:
    rows = []
    for key, df in preds.items():
        m = metrics(df.y_true, df.y_pred)
        t = METRICS / f"timing_{key}.json"
        timing = load_json(t) if t.exists() else {}
        m["Training time"] = timing.get("train_seconds")
        m["Inference time per CV"] = timing.get("infer_seconds_per_cv")
        rows.append({"Model": MODELS[key], **m})
    return pd.DataFrame(rows)


def main():
    preds = load_preds()
    if not preds:
        raise SystemExit("No predictions found in output/metrics/")
    table = comparison_table(preds)
    table.to_csv(TABLES / "model_comparison.csv", index=False)
    print(table.round(3).to_string(index=False))

    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})[["id", "occupation", "sex", "race"]]
    long = pd.concat([df.assign(model=MODELS[k]) for k, df in preds.items()]).merge(meta, on="id")
    long["abs_err"] = (clip_years(long.y_pred) - long.y_true).abs()
    long["err"] = clip_years(long.y_pred) - long.y_true
    long["band"] = to_band(long.y_true)
    (long.pivot_table(index="band", columns="model", values="abs_err", aggfunc="mean")
         .reindex(BAND_LABELS).to_csv(TABLES / "mae_by_band.csv"))
    long.pivot_table(index="occupation", columns="model", values="abs_err",
                     aggfunc="mean").to_csv(TABLES / "mae_by_occupation.csv")
    (long.pivot_table(index="band", columns="model", values="err", aggfunc="mean")
         .reindex(BAND_LABELS).to_csv(TABLES / "mean_error_by_band.csv"))

    for key, df in preds.items():
        cm = confusion_matrix(to_band(df.y_true), to_band(df.y_pred), labels=BAND_LABELS)
        pd.DataFrame(cm, index=BAND_LABELS, columns=BAND_LABELS).to_csv(TABLES / f"confusion_{key}.csv")

    if "bert" in preds:
        y = preds["bert"].y_true.to_numpy(float)
        rows = []
        for key, df in preds.items():
            if key == "bert":
                continue
            assert (df.id.values == preds["bert"].id.values).all()
            rows.append({"Baseline": MODELS[key],
                         **paired_bootstrap(y, preds["bert"].y_pred.values, df.y_pred.values)})
        gain = pd.DataFrame(rows)
        gain.to_csv(TABLES / "bert_gain_bootstrap.csv", index=False)
        print("\nBERT MAE gain (baseline MAE - BERT MAE), 95% paired bootstrap CI:")
        print(gain.round(3).to_string(index=False))

        b = long[long.model == MODELS["bert"]]
        fair = []
        for attr in ("sex", "race"):
            g = b.groupby(attr).agg(n=("err", "size"), MAE=("abs_err", "mean"),
                                    mean_error=("err", "mean"), mean_true=("y_true", "mean"))
            fair.append(g.reset_index().rename(columns={attr: "group"}).assign(attribute=attr))
        pd.concat(fair).to_csv(TABLES / "fairness_bert.csv", index=False)

        top = b.sort_values("abs_err", ascending=False).head(10)
        top[["id", "occupation", "y_true", "y_pred", "err"]].to_csv(
            TABLES / "bert_top10_errors.csv", index=False)


if __name__ == "__main__":
    main()
