"""Figures for the README, the video and the Streamlit app (output/figures/).

Usage:
    python -m src.visualization
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

from src.evaluation import MODELS, load_preds
from src.utils import BAND_LABELS, CAP, FIGURES, OCCUPATIONS, PROCESSED, RAW_IMAGES, TABLES, clip_years, load_json, OCR_DIR

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Median": "#9aa0a6", "TF-IDF + Ridge": "#e8a33d", "Embeddings + Ridge": "#5aa469",
           "Date rules (bonus)": "#8e6cc0", "Fine-tuned BERT": "#3b6fb6"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=130)
    plt.close(fig)


def target_distribution():
    df = pd.read_csv(PROCESSED / "cvs.csv", dtype={"id": str})
    df["Occupation"] = df.occupation.map(OCCUPATIONS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    sns.histplot(df.y, bins=np.arange(0, CAP + 1.01, 1), ax=axes[0], color="#3b6fb6")
    axes[0].set(title=f"Years of experience (n={len(df)}, {int((df.exp_year >= CAP).sum())} at cap {CAP})",
                xlabel="years (clipped 0-20)", ylabel="CVs")
    sns.boxplot(data=df, y="Occupation", x="y", ax=axes[1], color="#cfdcf0")
    axes[1].set(title="By occupation", xlabel="years", ylabel="")
    _save(fig, "target_distribution")


def ocr_quality():
    q = pd.read_csv(TABLES / "ocr_quality_per_cv.csv", dtype={"id": str})
    q["Occupation"] = q.occupation.map(OCCUPATIONS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(q.cer * 100, bins=40, ax=axes[0], color="#3b6fb6")
    axes[0].set(title=f"CER per CV (mean {q.cer.mean() * 100:.2f}%)", xlabel="CER %")
    sns.histplot(q.wer * 100, bins=40, ax=axes[1], color="#e8a33d")
    axes[1].set(title=f"WER per CV (mean {q.wer.mean() * 100:.2f}%)", xlabel="WER %")
    g = q.groupby("Occupation").apply(lambda d: d.years_ok.sum() / d.years_ref.sum() * 100,
                                      include_groups=False).sort_values()
    axes[2].barh(g.index, g.values, color="#5aa469")
    axes[2].set(title="Reference years read correctly", xlabel="%", xlim=(min(90, g.min() - 1), 100))
    for i, v in enumerate(g.values):
        axes[2].text(v, i, f" {v:.2f}%", va="center", fontsize=9)
    _save(fig, "ocr_quality")


def ocr_example(cv_id: str | None = None):
    """CV image with its PaddleOCR boxes, used in the app and the video."""
    if cv_id is None:
        ids = sorted(p.stem for p in OCR_DIR.glob("*.json"))
        cv_id = next(i for i in ids if (RAW_IMAGES / f"{i}.png").exists())
    j = load_json(OCR_DIR / f"{cv_id}.json")
    img = Image.open(RAW_IMAGES / f"{cv_id}.png").convert("RGB")
    draw = ImageDraw.Draw(img)
    for l in j["lines"]:
        color = (59, 111, 182) if l["score"] >= 0.8 else (214, 69, 65)
        draw.rectangle(l["box"], outline=color, width=2)
    img.save(FIGURES / "ocr_example_boxes.png")
    return cv_id


def pred_vs_true(preds):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    for ax, (key, df) in zip(np.atleast_1d(axes), preds.items()):
        name = MODELS[key]
        ax.scatter(df.y_true, clip_years(df.y_pred), s=8, alpha=0.45, color=PALETTE[name])
        ax.plot([0, CAP], [0, CAP], "k--", lw=1)
        mae = np.abs(clip_years(df.y_pred) - df.y_true).mean()
        ax.set(title=f"{name}\nMAE {mae:.2f}", xlabel="true years", xlim=(-0.5, CAP + .5), ylim=(-0.5, CAP + .5))
    np.atleast_1d(axes)[0].set_ylabel("predicted years")
    _save(fig, "pred_vs_true_all")
    for key, df in preds.items():  # one per model for the app
        fig, ax = plt.subplots(figsize=(5, 5))
        name = MODELS[key]
        ax.scatter(df.y_true, clip_years(df.y_pred), s=10, alpha=0.5, color=PALETTE[name])
        ax.plot([0, CAP], [0, CAP], "k--", lw=1)
        ax.set(title=name, xlabel="true years", ylabel="predicted years", xlim=(-0.5, CAP + .5), ylim=(-0.5, CAP + .5))
        _save(fig, f"pred_vs_true_{key}")


def mae_by_group():
    for fname, title in (("mae_by_band", "MAE by true experience band"),
                         ("mae_by_occupation", "MAE by occupation")):
        t = pd.read_csv(TABLES / f"{fname}.csv", index_col=0)
        if fname == "mae_by_occupation":
            t.index = t.index.map(OCCUPATIONS)
        cols = [c for c in PALETTE if c in t.columns]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        t[cols].plot.bar(ax=ax, color=[PALETTE[c] for c in cols], width=0.8)
        ax.set(title=title, ylabel="MAE (years)", xlabel="")
        ax.tick_params(axis="x", rotation=0 if fname == "mae_by_band" else 12)
        _save(fig, fname)


def confusion(preds):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.6))
    for ax, key in zip(np.atleast_1d(axes), preds):
        cm = pd.read_csv(TABLES / f"confusion_{key}.csv", index_col=0)
        norm = cm.div(cm.sum(1), axis=0)
        sns.heatmap(norm, annot=cm.values, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax)
        ax.set(title=MODELS[key], xlabel="predicted band", ylabel="true band")
    _save(fig, "confusion_bands")


def bert_gain():
    p = TABLES / "bert_gain_bootstrap.csv"
    if not p.exists():
        return
    g = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(g))
    ax.errorbar(g.gain, y, xerr=[g.gain - g.ci_low, g.ci_high - g.gain], fmt="o", color="#3b6fb6", capsize=4)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, g.Baseline)
    ax.set(title="BERT MAE reduction vs each baseline (95% paired bootstrap CI)",
           xlabel="baseline MAE - BERT MAE (years); > 0 means BERT is better")
    _save(fig, "bert_gain")


def main():
    target_distribution()
    if (TABLES / "ocr_quality_per_cv.csv").exists():
        ocr_quality()
    try:
        ocr_example()
    except StopIteration:
        pass
    preds = load_preds()
    if preds:
        pred_vs_true(preds)
        mae_by_group()
        confusion(preds)
        bert_gain()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
