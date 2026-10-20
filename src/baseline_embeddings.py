"""Baseline 2 - frozen sentence embeddings (BAAI/bge-small-en-v1.5) + Ridge.

bge-small accepts at most 512 tokens. Long CVs are split into consecutive
chunks of <= 510 word-piece tokens; each chunk is encoded and the CV vector is
the mean of its (L2-normalised) chunk vectors, so no part of the CV - and no
date - is dropped. The encoder is never fine-tuned; only Ridge is trained, with
alpha tuned on the validation set.

Usage:
    python -m src.baseline_embeddings
"""
from __future__ import annotations

import hashlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.utils import CAP, METRICS, MODELS, clip_years, load_split, save_json

ENCODER = "BAAI/bge-small-en-v1.5"
CHUNK = 510
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER, device="cpu")
        _encoder.max_seq_length = 512
    return _encoder


def chunk_text(text: str, tok) -> list[str]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return [""]
    return [tok.decode(ids[i:i + CHUNK]) for i in range(0, len(ids), CHUNK)]


def embed(texts, batch_size=32, cache_name: str | None = None) -> np.ndarray:
    texts = list(texts)
    cache = None
    if cache_name:
        h = hashlib.md5("\x00".join(texts).encode()).hexdigest()[:10]
        cache = MODELS / "baselines" / f"emb_{cache_name}_{h}.npy"
        if cache.exists():
            return np.load(cache)
    enc = get_encoder()
    chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in chunk_text(t, enc.tokenizer):
            chunks.append(c)
            owner.append(i)
    vecs = enc.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                      show_progress_bar=len(chunks) > 200)
    owner = np.array(owner)
    out = np.stack([vecs[owner == i].mean(0) for i in range(len(texts))])
    if cache is not None:
        np.save(cache, out)
    return out


def train(train_df, val_df, text_col="text", verbose=True):
    Xtr = embed(train_df[text_col], cache_name=f"train_{text_col}")
    Xva = embed(val_df[text_col], cache_name=f"val_{text_col}")
    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, train_df["y"] / CAP)
        mae = np.abs(clip_years(m.predict(Xva) * CAP) - val_df["y"]).mean()
        if verbose:
            print(f"  alpha={a:<5} val MAE={mae:.3f}")
        if best is None or mae < best[0]:
            best = (mae, a, m)
    model = best[2]
    model.best_alpha_, model.val_mae_ = best[1], best[0]
    return model


def predict(model, texts, cache_name=None) -> np.ndarray:
    return clip_years(model.predict(embed(texts, cache_name=cache_name)) * CAP)


def main():
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    t0 = time.perf_counter()
    model = train(tr, va)
    train_s = time.perf_counter() - t0  # includes encoding the training CVs
    joblib.dump(model, MODELS / "baselines" / "embeddings_ridge.joblib")
    t0 = time.perf_counter()
    pred = predict(model, te["text"])  # no cache -> honest inference time
    infer = (time.perf_counter() - t0) / len(te)
    pd.DataFrame({"id": te.id, "y_true": te.y, "y_pred": pred}).to_csv(
        METRICS / "preds_embeddings.csv", index=False)
    pd.DataFrame({"id": va.id, "y_true": va.y,
                  "y_pred": predict(model, va["text"], cache_name="val_text")}).to_csv(
        METRICS / "preds_embeddings_val.csv", index=False)
    save_json({"train_seconds": train_s, "infer_seconds_per_cv": infer, "encoder": ENCODER,
               "best_alpha": model.best_alpha_, "val_mae": model.val_mae_},
              METRICS / "timing_embeddings.json")
    print(f"best alpha={model.best_alpha_}; test MAE = {np.abs(pred - te.y).mean():.3f}")


if __name__ == "__main__":
    main()
