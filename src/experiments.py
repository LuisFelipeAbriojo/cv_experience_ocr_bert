"""Experiments for the analytical questions (Q1 probe, Q2 OCR noise, Q3 sensitivity).

  probe      Q1 - 20 test CVs edited three ways; does each prediction move like a
             recruiter's would?
               shift +5   every year +5            -> expected change   0
               drop job   remove the latest job    -> expected change  -its duration
               earlier    first job starts 5y earlier -> expected change +5 (if no overlap)
  ocr_noise  Q2 - every trained model on OCR text vs reference text of the same test CVs
  cleaning   Q3 - OCR cleaning variant (raw / cleaned / low-confidence removed) for the
             models that are cheap to retrain, plus TF-IDF with/without digits
  collect    Q3 - gathers BERT variant runs (Huber loss, raw OCR text, ...) into one table

Usage:
    python -m src.experiments probe|ocr_noise|cleaning|collect
"""
from __future__ import annotations

import re
import sys

import joblib
import numpy as np
import pandas as pd

from src import baseline_embeddings, baseline_median, baseline_rules, baseline_tfidf
from src.baseline_rules import RANGE, total_years
from src.utils import CAP, METRICS, MODELS, SEED, TABLES, clip_years, load_json, load_split

JOB_LINE = re.compile(r"(?m)^.*(?:19|20)\d\d\s*\.\s*\d{1,2}\s*-\s*(?:(?:19|20)\d\d\s*\.\s*\d{1,2}|present).*$")


def load_models():
    """Return {name: callable(texts) -> predictions} for every trained model."""
    out = {}
    med = load_json(MODELS / "baselines" / "median.json")["median"]
    out["Median"] = lambda t: baseline_median.predict(med, t)
    tf = joblib.load(MODELS / "baselines" / "tfidf.joblib")
    out["TF-IDF + Ridge"] = lambda t: baseline_tfidf.predict(tf, t)
    emb = joblib.load(MODELS / "baselines" / "embeddings_ridge.joblib")
    out["Embeddings + Ridge"] = lambda t: baseline_embeddings.predict(emb, t)
    out["Date rules (bonus)"] = lambda t: baseline_rules.predict(med, t)
    bert_dir = MODELS / "bert" / "bert_main"
    if (bert_dir / "config_run.json").exists():
        from src import train_bert

        bundle = train_bert.load(bert_dir)
        out["Fine-tuned BERT"] = lambda t: train_bert.predict(bundle, t)
    return out


# --------------------------------------------------------------------------- #
# Q1 probe
# --------------------------------------------------------------------------- #
def shift_years(text: str, k: int = 5) -> str:
    return re.sub(r"(?<!\d)((?:19|20)\d\d)(?!\d)", lambda m: str(int(m.group(1)) + k), text)


def drop_latest_job(text: str) -> tuple[str, float]:
    """Remove the first job block (title/date line + its bullets)."""
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if JOB_LINE.match(l)]
    if len(idx) < 2:
        return text, np.nan
    i = idx[0]
    j = i + 1
    while j < len(lines) and lines[j].lstrip().startswith(("-", "•", "*")):
        j += 1
    new = "\n".join(lines[:i] + lines[j:])
    return new, total_years(new) - total_years(text)


def start_earlier(text: str, k: int = 5) -> tuple[str, float]:
    """Move the start of the earliest job k years back."""
    ms = list(RANGE.finditer(text))
    if not ms:
        return text, np.nan
    first = min(ms, key=lambda m: (int(m.group(1)), int(m.group(2))))
    a, b = first.span(1)
    new = text[:a] + str(int(first.group(1)) - k) + text[b:]
    return new, total_years(new) - total_years(text)


def probe(n: int = 20):
    te = load_split("test")
    # 20 CVs with >= 2 jobs, stratified over the target range
    te = te[te.text.map(lambda t: len(JOB_LINE.findall(t)) >= 2)]
    sample = te.sort_values("y").iloc[np.linspace(0, len(te) - 1, n).astype(int)]
    variants = {"original": (sample.text.tolist(), np.zeros(n)),
                "shift +5 years": ([shift_years(t) for t in sample.text], np.zeros(n))}
    dropped = [drop_latest_job(t) for t in sample.text]
    variants["drop latest job"] = ([d[0] for d in dropped], np.array([d[1] for d in dropped]))
    earlier = [start_earlier(t) for t in sample.text]
    variants["first job 5y earlier"] = ([e[0] for e in earlier], np.array([e[1] for e in earlier]))

    models = load_models()
    rows = []
    for name, f in models.items():
        base = f(variants["original"][0])
        for v, (texts, expected) in variants.items():
            if v == "original":
                continue
            delta = f(texts) - base
            # expected change is what the true answer would become (clipped to [0, CAP])
            exp_delta = clip_years(sample.y.values + expected) - sample.y.values
            ok = ~np.isnan(exp_delta)
            rows.append(dict(model=name, edit=v, expected_change=np.nanmean(exp_delta),
                             mean_change=delta[ok].mean(),
                             mean_abs_deviation=np.abs(delta[ok] - exp_delta[ok]).mean(),
                             moved_right_direction=np.mean(np.sign(np.round(delta[ok], 1)) ==
                                                           np.sign(np.round(exp_delta[ok], 1)))))
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "probe_q1.csv", index=False)
    sample[["id", "y"]].to_csv(TABLES / "probe_q1_ids.csv", index=False)
    print(res.round(3).to_string(index=False))


# --------------------------------------------------------------------------- #
# Q2 OCR noise
# --------------------------------------------------------------------------- #
def ocr_noise():
    te = load_split("test")
    rows = []
    for name, f in load_models().items():
        e_ocr = np.abs(f(te.text) - te.y).mean()
        e_ref = np.abs(f(te.text_ref) - te.y).mean()
        rows.append({"model": name, "MAE OCR text": e_ocr, "MAE reference text": e_ref,
                     "gap (OCR - ref)": e_ocr - e_ref,
                     "relative gap %": 100 * (e_ocr - e_ref) / max(e_ref, 1e-9)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "ocr_noise_q2.csv", index=False)
    print(res.round(3).to_string(index=False))


# --------------------------------------------------------------------------- #
# Q3 sensitivity
# --------------------------------------------------------------------------- #
def cleaning():
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    rows = []
    for col, label in (("text_raw", "raw OCR"), ("text", "cleaned OCR"),
                       ("text_lowconf", "cleaned, low-conf lines removed")):
        med = baseline_median.train(tr)
        rows.append(dict(setup=label, model="Median", MAE=np.abs(baseline_median.predict(med, te[col]) - te.y).mean()))
        rows.append(dict(setup=label, model="Date rules (bonus)",
                         MAE=np.abs(baseline_rules.predict(med, te[col]) - te.y).mean()))
        tf = baseline_tfidf.train(tr, va, text_col=col, verbose=False)
        rows.append(dict(setup=label, model="TF-IDF + Ridge",
                         MAE=np.abs(baseline_tfidf.predict(tf, te[col]) - te.y).mean()))
        emb = baseline_embeddings.train(tr, va, text_col=col, verbose=False)
        rows.append(dict(setup=label, model="Embeddings + Ridge",
                         MAE=np.abs(baseline_embeddings.predict(emb, te[col], cache_name=f"test_{col}") - te.y).mean()))
        print(rows[-4:], flush=True)
    for keep in (True, False):
        tf = baseline_tfidf.train(tr, va, keep_digits=keep, verbose=False)
        rows.append(dict(setup="TF-IDF digits kept" if keep else "TF-IDF digits removed",
                         model="TF-IDF + Ridge", MAE=np.abs(baseline_tfidf.predict(tf, te.text) - te.y).mean()))
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "sensitivity_cleaning_q3.csv", index=False)
    print(res.pivot_table(index="model", columns="setup", values="MAE").round(3).to_string())


def collect():
    """BERT variants trained with src.train_bert (each writes preds_<run>.csv)."""
    rows = []
    for p in sorted(METRICS.glob("timing_bert*.json")):
        key = p.stem.replace("timing_", "")
        pr = METRICS / f"preds_{key}.csv"
        if not pr.exists():
            continue
        d = pd.read_csv(pr)
        cfg = load_json(p)
        rows.append(dict(run=key, loss=cfg.get("loss"), strategy=cfg.get("strategy"),
                         max_len=cfg.get("max_len"), text_col=cfg.get("text_col", "text"),
                         test_MAE=np.abs(clip_years(d.y_pred) - d.y_true).mean(),
                         best_val_MAE=cfg.get("best_val_mae"),
                         train_minutes=cfg.get("train_seconds", 0) / 60))
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "sensitivity_bert_q3.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    {"probe": probe, "ocr_noise": ocr_noise, "cleaning": cleaning, "collect": collect}[sys.argv[1]]()
