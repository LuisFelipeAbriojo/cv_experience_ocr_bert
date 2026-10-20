"""Stage 5 - Fine-tune a lightweight BERT-style encoder as a regressor.

AutoModelForSequenceClassification(num_labels=1, problem_type="regression")
predicts y / CAP. Loss is MSE (default) or Huber. Early stopping on validation
MAE (in years). Long CVs are handled by one of three strategies:

  head       first max_len tokens
  head_tail  first `head` tokens + last (max_len - head) tokens
  chunk      split the CV into windows of max_len tokens (stride 64); every
             window is a training example with the CV's label and the CV
             prediction is the mean of its windows

Usage:
    python -m src.train_bert --run bert_main
    python -m src.train_bert --run bert_huber --loss huber
"""
from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.utils import CAP, METRICS, MODELS, SEED, clip_years, load_json, load_split, save_json, set_seed

# PopResume CVs are short (cleaned OCR: max ~190 word-pieces), so max_len=256 keeps
# every token - and every date - while halving CPU time versus 512.
DEFAULTS = dict(model_name="distilbert-base-uncased", max_len=256, strategy="head_tail", head=128,
                loss="mse", huber_delta=0.1, lr=5e-5, batch_size=16, epochs=15, patience=3,
                warmup=0.1, weight_decay=0.01, seed=SEED, stride=64)


# --------------------------------------------------------------------------- #
def encode(texts, tok, cfg):
    """Return a list of token-id lists and, for every item, the index of its CV."""
    body = cfg["max_len"] - 2
    out, owner = [], []
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        if cfg["strategy"] == "head":
            pieces = [ids[:body]]
        elif cfg["strategy"] == "head_tail":
            h = cfg["head"]
            pieces = [ids if len(ids) <= body else ids[:h] + ids[-(body - h):]]
        elif cfg["strategy"] == "chunk":
            step = body - cfg["stride"]
            pieces = [ids[s:s + body] for s in range(0, max(len(ids) - cfg["stride"], 1), step)]
        else:
            raise ValueError(cfg["strategy"])
        for p in pieces:
            out.append([tok.cls_token_id, *p, tok.sep_token_id])
            owner.append(i)
    return out, np.array(owner)


def collate(batch_ids, pad_id):
    L = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for k, x in enumerate(batch_ids):
        ids[k, :len(x)] = torch.tensor(x)
        att[k, :len(x)] = 1
    return ids, att


@torch.no_grad()
def predict_encoded(model, enc, owner, n_cvs, pad_id, batch_size=32):
    model.eval()
    order = np.argsort([len(x) for x in enc])  # length bucketing -> less padding
    raw = np.zeros(len(enc))
    for s in range(0, len(enc), batch_size):
        idx = order[s:s + batch_size]
        ids, att = collate([enc[i] for i in idx], pad_id)
        raw[idx] = model(input_ids=ids, attention_mask=att).logits.squeeze(-1).numpy()
    per_cv = np.array([raw[owner == i].mean() for i in range(n_cvs)])
    return clip_years(per_cv * CAP)


def predict(bundle, texts, batch_size=32):
    model, tok, cfg = bundle
    enc, owner = encode(list(texts), tok, cfg)
    return predict_encoded(model, enc, owner, len(texts), tok.pad_token_id, batch_size)


def load(run_dir: Path):
    cfg = load_json(run_dir / "config_run.json")
    tok = AutoTokenizer.from_pretrained(run_dir)
    model = AutoModelForSequenceClassification.from_pretrained(run_dir)
    model.eval()
    return model, tok, cfg


# --------------------------------------------------------------------------- #
def train(train_df, val_df, cfg, out_dir: Path, text_col="text"):
    set_seed(cfg["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=1, problem_type="regression")
    tr_enc, tr_owner = encode(train_df[text_col].tolist(), tok, cfg)
    va_enc, va_owner = encode(val_df[text_col].tolist(), tok, cfg)
    y_tr = (train_df["y"].to_numpy(float) / CAP)[tr_owner]
    print(f"train examples={len(tr_enc)} (from {len(train_df)} CVs), val={len(va_enc)}", flush=True)

    loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else nn.HuberLoss(delta=cfg["huber_delta"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * math.ceil(len(tr_enc) / cfg["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * steps), steps)
    rng = np.random.default_rng(cfg["seed"])

    history, best, bad = [], math.inf, 0
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # shuffle, then bucket by length inside blocks of 50 batches (faster on CPU)
        perm = rng.permutation(len(tr_enc))
        blk = cfg["batch_size"] * 50
        perm = np.concatenate([sorted(perm[s:s + blk], key=lambda i: len(tr_enc[i]))
                               for s in range(0, len(perm), blk)])
        batches = [perm[s:s + cfg["batch_size"]] for s in range(0, len(perm), cfg["batch_size"])]
        rng.shuffle(batches)
        t_ep, run_loss = time.time(), 0.0
        for k, idx in enumerate(batches, 1):
            ids, att = collate([tr_enc[i] for i in idx], tok.pad_token_id)
            out = model(input_ids=ids, attention_mask=att).logits.squeeze(-1)
            loss = loss_fn(out, torch.tensor(y_tr[idx], dtype=torch.float))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += loss.item()
            if k % 20 == 0:
                print(f"  epoch {epoch} step {k}/{len(batches)} loss={run_loss / k:.4f} "
                      f"({(time.time() - t_ep) / k:.1f}s/step)", flush=True)
        val_pred = predict_encoded(model, va_enc, va_owner, len(val_df), tok.pad_token_id)
        val_mae = float(np.abs(val_pred - val_df["y"].to_numpy(float)).mean())
        history.append(dict(epoch=epoch, train_loss=run_loss / len(batches), val_mae=val_mae,
                            minutes=(time.time() - t_ep) / 60))
        print(f"epoch {epoch}: train_loss={run_loss / len(batches):.4f} val_MAE={val_mae:.3f} "
              f"({(time.time() - t_ep) / 60:.1f} min)", flush=True)
        if val_mae < best - 1e-4:
            best, bad = val_mae, 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("early stopping", flush=True)
                break
    train_s = time.time() - t_start
    save_json(cfg, out_dir / "config_run.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return train_s, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bert_main")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--max-train", type=int, help="subsample training CVs (smoke tests)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    out_dir = MODELS / "bert" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    if a.max_train:
        tr = tr.sample(a.max_train, random_state=SEED)
    train_s, best_val, history = train(tr, va, cfg, out_dir, a.text_col)

    bundle = load(out_dir)
    t0 = time.perf_counter()
    pred = predict(bundle, te[a.text_col])
    infer = (time.perf_counter() - t0) / len(te)
    key = "bert" if a.run == "bert_main" else a.run
    pd.DataFrame({"id": te.id, "y_true": te.y, "y_pred": pred}).to_csv(
        METRICS / f"preds_{key}.csv", index=False)
    pd.DataFrame({"id": va.id, "y_true": va.y, "y_pred": predict(bundle, va[a.text_col])}).to_csv(
        METRICS / f"preds_{key}_val.csv", index=False)
    save_json({"train_seconds": train_s, "infer_seconds_per_cv": infer, "best_val_mae": best_val,
               "epochs_run": len(history), "hardware": f"{platform.processor()} (CPU only)",
               "torch_threads": torch.get_num_threads(), "text_col": a.text_col, **cfg},
              METRICS / f"timing_{key}.json")
    print(f"test MAE = {np.abs(pred - te.y).mean():.3f}; train {train_s / 60:.1f} min")


if __name__ == "__main__":
    main()
