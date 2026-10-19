"""Shared paths, constants and helpers used by every stage of the pipeline."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_IMAGES = RAW / "images"
OCR_DIR = DATA / "ocr"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
METRICS = OUTPUT / "metrics"

OCCUPATIONS = {
    "registered_nurses": "Registered nurses",
    "elementary_middle_school_teachers": "Elementary / middle school teachers",
    "software_developers": "Software developers",
    "accountants_auditors": "Accountants & auditors",
    "construction_laborers": "Construction laborers",
}

CAP = 20  # exp_year is clipped to [0, CAP]
BANDS = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, CAP, f"11-{CAP}")]
BAND_LABELS = [b[2] for b in BANDS]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (RAW_IMAGES, OCR_DIR, PROCESSED, SPLITS, MODELS / "baselines",
              MODELS / "bert", FIGURES, TABLES, METRICS):
        d.mkdir(parents=True, exist_ok=True)


def clip_years(y, cap: int = CAP):
    return np.clip(np.asarray(y, dtype=float), 0, cap)


def to_band(y) -> np.ndarray:
    """Map (rounded) years to the four experience bands used for evaluation."""
    y = np.rint(clip_years(y)).astype(int)
    out = np.empty(len(y), dtype=object)
    for lo, hi, name in BANDS:
        out[(y >= lo) & (y <= hi)] = name
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_split(name: str):
    """Return the processed dataframe restricted to one split (train/val/test)."""
    import pandas as pd

    df = pd.read_csv(PROCESSED / "cvs.csv", dtype={"id": str})
    ids = pd.read_csv(SPLITS / f"{name}.csv", dtype={"id": str})["id"]
    return df.set_index("id").loc[ids].reset_index()
