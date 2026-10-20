"""Stage 3 - Stratified train / validation / test split (70 / 15 / 15).

Split is done by CV id, stratified jointly by occupation and experience band,
with a fixed seed. Every model reads these three files, so all differences in
results come from the model and not from the data.

Usage:
    python -m src.split
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import PROCESSED, SEED, SPLITS, TABLES, ensure_dirs, to_band


def main():
    ensure_dirs()
    df = pd.read_csv(PROCESSED / "cvs.csv", dtype={"id": str})
    strata = df["occupation"] + "|" + to_band(df["y"])
    counts = strata.map(strata.value_counts())
    strata = strata.where(counts >= 7, df["occupation"])  # tiny strata -> occupation only
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50,
                                 stratify=strata.loc[rest.index], random_state=SEED)
    summary = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[["id", "occupation", "y"]].to_csv(SPLITS / f"{name}.csv", index=False)
        summary.append(dict(split=name, n=len(part), mean_y=part.y.mean(), median_y=part.y.median(),
                            **part["occupation"].value_counts().to_dict()))
    s = pd.DataFrame(summary)
    s.to_csv(TABLES / "split_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
