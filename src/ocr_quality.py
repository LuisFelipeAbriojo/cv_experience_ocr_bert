"""Stage 2a - OCR quality against the PopResume reference text.

Both texts are normalised the same way (lower-case, unified dashes/quotes,
collapsed whitespace) and compared with jiwer:
  * CER / WER          - order-sensitive edit distances.
  * bag-of-words F1    - order-free, separates misreads from reading-order effects.
  * year recall        - share of 4-digit years (19xx/20xx) in the reference whose
                         digits appear in the OCR text (multiset match), and the
                         share of CVs where every year was read correctly.
  * date-range recall  - same, for whole "YYYY.MM - YYYY.MM" ranges, as read and
                         after re-inserting a dash that OCR dropped
                         ("2017.102020.06"); `dash_dropped_share` counts those.

The reference text is used ONLY here (and for the OCR-noise experiment in
Question 2); it is never a model input.

Usage:
    python -m src.ocr_quality
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import jiwer
import numpy as np
import pandas as pd

from src.ocr_extraction import ordered_text
from src.utils import METRICS, OCR_DIR, RAW, TABLES, load_json, save_json

YEAR = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)(?!\d)")
# PopResume job dates look like "2017.10–2020.06"
RANGE = re.compile(r"(?<!\d)((?:19|20)\d\d\.\d\d)\s*-\s*((?:19|20)\d\d\.\d\d|present)", re.I)
GLUED = re.compile(r"((?:19|20)\d\d\.\d\d)(?=(?:19|20)\d\d\.\d\d)")


def split_glued(text: str) -> str:
    """'2017.102020.06' (dash dropped by OCR) -> '2017.10 - 2020.06'."""
    return GLUED.sub(r"\1 - ", text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = text.lower()
    text = re.sub(r"\s*-\s*", " - ", text)
    return re.sub(r"\s+", " ", text).strip()


def bow_f1(ref: str, hyp: str) -> float:
    r, h = Counter(ref.split()), Counter(hyp.split())
    tp = sum((r & h).values())
    if tp == 0:
        return 0.0
    p, q = tp / sum(h.values()), tp / sum(r.values())
    return 2 * p * q / (p + q)


def multiset_recall(ref_items, hyp_items):
    r, h = Counter(ref_items), Counter(hyp_items)
    total = sum(r.values())
    return (sum((r & h).values()), total)


def ranges(text: str):
    return [f"{a}-{b.lower()}" for a, b in RANGE.findall(text)]


def years(text: str):
    # years are counted on digits only: a dropped dash is reported separately
    return YEAR.findall(split_glued(text).replace(".", " "))


def evaluate(ocr_texts: dict[str, str], ref: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cv_id, ref_text in zip(ref["id"], ref["text"]):
        if cv_id not in ocr_texts:
            continue
        r, h = normalize(ref_text), normalize(ocr_texts[cv_id])
        yr_ok, yr_tot = multiset_recall(years(r), years(h))
        rg_ok, rg_tot = multiset_recall(ranges(r), ranges(h))
        rg_fix_ok, _ = multiset_recall(ranges(r), ranges(split_glued(h)))
        rows.append(dict(
            id=cv_id, cer=jiwer.cer(r, h) if h else 1.0, wer=jiwer.wer(r, h) if h else 1.0,
            bow_f1=bow_f1(r, h), years_ref=yr_tot, years_ok=yr_ok,
            ranges_ref=rg_tot, ranges_ok=rg_ok, ranges_ok_after_dash_fix=rg_fix_ok,
            dash_dropped=len(GLUED.findall(h)),
            all_years_ok=yr_ok == yr_tot,
        ))
    return pd.DataFrame(rows)


def main():
    ref = pd.read_csv(RAW / "reference_text.csv", dtype={"id": str})
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})[["id", "occupation"]]
    ocr = {}
    for p in OCR_DIR.glob("*.json"):
        j = load_json(p)
        ocr[j["id"]] = ordered_text(j)
    df = evaluate(ocr, ref).merge(meta, on="id")
    df.to_csv(TABLES / "ocr_quality_per_cv.csv", index=False)

    def agg(g):
        return pd.Series(dict(
            n=len(g), cer_mean=g.cer.mean(), cer_median=g.cer.median(),
            wer_mean=g.wer.mean(), wer_median=g.wer.median(), bow_f1_mean=g.bow_f1.mean(),
            year_recall=g.years_ok.sum() / max(g.years_ref.sum(), 1),
            range_recall=g.ranges_ok.sum() / max(g.ranges_ref.sum(), 1),
            range_recall_after_dash_fix=g.ranges_ok_after_dash_fix.sum() / max(g.ranges_ref.sum(), 1),
            dash_dropped_share=g.dash_dropped.sum() / max(g.ranges_ref.sum(), 1),
            cvs_all_years_ok=g.all_years_ok.mean()))

    by_occ = df.groupby("occupation").apply(agg, include_groups=False)
    overall = agg(df)
    by_occ.loc["ALL"] = overall
    by_occ.to_csv(TABLES / "ocr_quality_by_occupation.csv")
    save_json(overall.to_dict(), METRICS / "ocr_quality.json")
    print(by_occ.round(4).to_string())


if __name__ == "__main__":
    main()
