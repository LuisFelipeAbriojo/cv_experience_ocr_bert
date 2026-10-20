"""Stage 2b - Cleaning, reading order, PII removal and masking of direct answers.

Input: cached OCR JSONs (data/ocr). Output: data/processed/cvs.csv with
  id, occupation, exp_year, y (clipped target), sex, race (fairness check only),
  text            cleaned OCR text            <- the model input
  text_raw        OCR in reading order, only PII removal + masking (Q3)
  text_lowconf    cleaned OCR without lines below MIN_CONF (Q3)
  text_ref        reference text, cleaned identically (Q2 only, never trained on)

Cleaning steps (applied in this order):
  1. reading order from the boxes (src.ocr_extraction.reading_order)
  2. unify dashes (OCR returns "-", "~" or "—" for the en dash in "2014.01–2019.03"),
     re-insert the dash when OCR dropped it ("2014.012019.03"), and fix common
     OCR digit confusions inside date tokens (O->0, l/I->1, S->5)
  3. merge broken lines: hyphenated line ends and bullet continuations
  4. remove PII: everything above the first section header (name banner and the
     "STATE, USA | phone | email" line), plus any stray e-mail / phone / URL
  5. mask phrases that state the answer (total experience, age, date of birth)
     with "[MASKED]" and count the affected CVs

Usage:
    python -m src.preprocessing
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.ocr_extraction import ordered_text
from src.utils import CAP, METRICS, OCR_DIR, PROCESSED, RAW, ensure_dirs, load_json, save_json

MIN_CONF = 0.80  # threshold for the "low-confidence lines removed" variant (see README)

HEADERS = re.compile(r"(?im)^\s*(work\s*history|work\s*experience|experience|employment"
                     r"|professional\s*experience|education)\s*$")
EMAIL = re.compile(r"\S+@\S+\.\w+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
URL = re.compile(r"(https?://|www\.)\S+", re.I)
CONTACT_LINE = re.compile(r"(?i)\b[A-Z]{2}\s*,\s*USA\b")

# phrases that reveal the target directly
MASKS = {
    "total_experience": re.compile(
        r"(?i)\b(?:over|more than|nearly|almost|about|around)?\s*\d{1,2}(?:\.\d)?\+?\s*"
        r"(?:years?|yrs?)\.?\s*(?:of\s+)?(?:\w+\s+){0,3}?(?:experience|exp\.?)\b"
        r"|\b(?:a|one|two|three)\s+decades?\s+of\s+(?:\w+\s+){0,2}?experience\b"
        r"|\bexperience\s*[:\-]\s*\d{1,2}\+?\s*(?:years?|yrs?)\b"),
    "age": re.compile(r"(?i)\bage[d]?\s*[:\-]?\s*\d{2}\b|\b\d{2}[\s-]*(?:years?|yrs?)[\s-]*old\b"),
    "date_of_birth": re.compile(
        r"(?i)\b(?:date\s+of\s+birth|d\.?\s?o\.?\s?b\.?|birth\s*date|born(?:\s+(?:on|in))?)"
        r"\s*[:\-]?\s*[^\n|]{0,25}"),
}

_DATE_TOKEN = re.compile(r"[0-9OolIS]{4}\.[0-9OolIS]{2}")


def _fix_date_token(m: re.Match) -> str:
    return m.group(0).translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5"}))


def normalize_dates(text: str) -> str:
    text = re.sub(r"[–—−~]", "-", text)
    text = _DATE_TOKEN.sub(_fix_date_token, text)
    # PaddleOCR often drops the en dash: "2017.102020.06" -> "2017.10 - 2020.06"
    # (months always have two digits, so the split point is unambiguous)
    text = re.sub(r"((?:19|20)\d\d\.\d\d)\s*(?=(?:19|20)\d\d\.\d\d)", r"\1 - ", text)
    # "2014.01 - 2019.03" / "2014.01-2019.03" -> "2014.01 - 2019.03"
    return re.sub(r"(\d{4}\.\d{2})\s*-\s*(\d{4}\.\d{2}|present|current)", r"\1 - \2", text, flags=re.I)


def merge_lines(text: str) -> str:
    out: list[str] = []
    for line in (l.strip() for l in text.split("\n")):
        if not line:
            continue
        if out and out[-1].endswith("-") and line[:1].islower():
            out[-1] = out[-1][:-1] + line  # hyphenation across lines
        elif (out and line[:1].islower() and not HEADERS.match(line)
              and re.match(r"^[-•*]", out[-1])):
            out[-1] += " " + line  # bullet that wrapped to a new line
        else:
            out.append(line)
    return "\n".join(out)


def remove_pii(text: str) -> str:
    m = HEADERS.search(text)
    if m:
        text = text[m.start():]  # drops the name banner and contact line
    else:
        lines = text.split("\n")
        if lines and re.fullmatch(r"[A-Z][A-Z' .-]{2,40}", lines[0]):
            lines = lines[1:]
        text = "\n".join(l for l in lines if not CONTACT_LINE.search(l))
    text = EMAIL.sub(" ", text)
    text = URL.sub(" ", text)
    # phones, but never the date ranges ("2014.01 - 2019.03" also looks like digits)
    text = PHONE.sub(lambda m: m.group(0) if re.search(r"\d{4}\.\d{2}", m.group(0)) else " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def mask(text: str) -> tuple[str, dict]:
    hits = {}
    for name, pat in MASKS.items():
        text, n = pat.subn("[MASKED]", text)
        hits[name] = n
    return text, hits


def clean(text: str) -> tuple[str, dict]:
    text = normalize_dates(text)
    text = merge_lines(text)
    text = remove_pii(text)
    return mask(text)


def minimal(text: str) -> str:
    """Raw OCR in reading order: only PII removal and masking (names are never features)."""
    return mask(remove_pii(text))[0]


def main():
    ensure_dirs()
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    ref = pd.read_csv(RAW / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]
    rows, mask_log = [], []
    for r in meta.itertuples():
        p = OCR_DIR / f"{r.id}.json"
        if not p.exists():
            continue
        j = load_json(p)
        if not j["ok"] or not j["lines"]:
            continue
        raw = ordered_text(j)
        text, hits = clean(raw)
        text_low, _ = clean(ordered_text(j, min_score=MIN_CONF))
        text_ref, _ = clean(ref.get(r.id, ""))
        if sum(hits.values()):
            mask_log.append({"id": r.id, **hits})
        rows.append(dict(id=r.id, occupation=r.occupation, exp_year=r.exp_year,
                         y=float(np.clip(r.exp_year, 0, CAP)), sex=r.sex, race=r.race,
                         text=text, text_raw=minimal(raw), text_lowconf=text_low,
                         text_ref=text_ref, n_chars=len(text)))
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "cvs.csv", index=False)
    log = pd.DataFrame(mask_log, columns=["id", *MASKS])
    log.to_csv(PROCESSED / "masking_log.csv", index=False)
    stats = dict(n_cvs=len(df), n_cvs_masked=len(log),
                 masked_by_type={k: int((log[k] > 0).sum()) for k in MASKS},
                 n_at_cap=int((df.exp_year >= CAP).sum()),
                 share_at_cap=float((df.exp_year >= CAP).mean()),
                 min_conf_threshold=MIN_CONF)
    save_json(stats, METRICS / "preprocessing_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
