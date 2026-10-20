"""Stage 0 - Stratified sampling of no-photo CV images from PopResume.

For each occupation we stream its image shards, keep only rows with
has_photo == False, shuffle with a fixed seed and take N rows. Each image is
saved as data/raw/images/<id>.png and the attributes we are allowed to keep
(occupation, target, and sex/race for the fairness check only) go to
data/raw/metadata.csv. The reference text (used ONLY to measure OCR error) is
stored in data/raw/reference_text.csv.

Usage:
    python -m src.download_data --per-occupation 800
"""
from __future__ import annotations

import argparse
import io
import time

import pandas as pd
from datasets import Image, load_dataset
from PIL import Image as PILImage

from src.utils import OCCUPATIONS, RAW, RAW_IMAGES, SEED, ensure_dirs

REPO = "hf://datasets/sumin-yu/PopResume"
# Columns kept from the image rows. age, names, state and region are dropped on
# purpose so they can never leak into a model.
KEEP = ["id", "job", "sex", "race", "exp_year", "edu_level"]


def sample_occupation(slug: str, n: int, seed: int) -> list[dict]:
    ds = load_dataset(
        "parquet",
        data_files=f"{REPO}/data/images/{slug}-*.parquet",
        split="train",
        streaming=True,
    )
    # keep images as compressed bytes in the shuffle buffer (5000 decoded pages
    # would need ~30 GB of RAM); only the sampled rows are decoded and saved
    ds = ds.cast_column("image", Image(decode=False))
    ds = ds.filter(lambda has_photo: not has_photo, input_columns=["has_photo"])
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    rows = []
    t0 = time.time()
    for r in ds.take(n):
        path = RAW_IMAGES / f"{r['id']}.png"
        img = PILImage.open(io.BytesIO(r["image"]["bytes"]))
        if not path.exists():
            img.convert("RGB").save(path)
        rows.append({k: r.get(k) for k in KEEP} | {"occupation": slug,
                                                    "width": img.width, "height": img.height})
        if len(rows) % 100 == 0:
            print(f"  {slug}: {len(rows)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def fetch_reference_text(ids: set[str]) -> pd.DataFrame:
    parts = []
    for slug in OCCUPATIONS:
        t = pd.read_parquet(f"{REPO}/data/text/{slug}.parquet", columns=["id", "text"])
        t["id"] = t["id"].astype(str)
        parts.append(t[t["id"].isin(ids)])
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-occupation", type=int, default=800)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--occupation", choices=list(OCCUPATIONS),
                    help="sample only this occupation (lets the 5 run in parallel)")
    args = ap.parse_args()
    ensure_dirs()

    slugs = [args.occupation] if args.occupation else list(OCCUPATIONS)
    for slug in slugs:
        part = RAW / f"metadata_{slug}.csv"
        if part.exists() and len(pd.read_csv(part)) >= args.per_occupation:
            print(f"{slug}: already sampled, skipping")
            continue
        print(f"Sampling {slug} ...", flush=True)
        pd.DataFrame(sample_occupation(slug, args.per_occupation, args.seed)).to_csv(part, index=False)
    if args.occupation:
        return

    meta = pd.concat([pd.read_csv(RAW / f"metadata_{s}.csv", dtype={"id": str}) for s in OCCUPATIONS],
                     ignore_index=True)
    meta_path = RAW / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    meta[["id", "occupation"]].to_csv(RAW / "sampled_ids.csv", index=False)

    ref = fetch_reference_text(set(meta["id"]))
    ref.to_csv(RAW / "reference_text.csv", index=False)
    print(f"Sampled {len(meta)} CVs; reference text for {len(ref)}")
    print(meta.groupby("occupation").size())


if __name__ == "__main__":
    main()
