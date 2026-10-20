"""Stage 1 - Text extraction with PaddleOCR.

Runs PaddleOCR once per image and caches one JSON per CV in data/ocr/<id>.json
with every line's text, confidence and bounding box, plus the timing. Already
processed images are skipped, so the job can be stopped and resumed. Use
--shard i --num-shards k to run k processes in parallel.

Reading order is rebuilt from the boxes by `reading_order` (see its docstring),
which also detects two-column layouts.

Usage:
    python -m src.ocr_extraction [--limit 20] [--shard 0 --num-shards 2]
    python -m src.ocr_extraction --summary
"""
from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np
import pandas as pd

from src.utils import METRICS, OCR_DIR, RAW, RAW_IMAGES, TABLES, ensure_dirs, load_json, save_json

DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "en_PP-OCRv5_mobile_rec"


def build_ocr(cpu_threads: int | None = None):
    from paddleocr import PaddleOCR

    kwargs = dict(
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if cpu_threads:
        kwargs["cpu_threads"] = cpu_threads
    return PaddleOCR(**kwargs)


def run_ocr(ocr, image) -> dict:
    """Run PaddleOCR on a path or numpy image and return a JSON-serialisable dict."""
    t0 = time.perf_counter()
    res = ocr.predict(image)[0]
    elapsed = time.perf_counter() - t0
    boxes = np.asarray(res["rec_boxes"]).reshape(-1, 4).tolist()
    lines = [
        {"text": t, "score": float(s), "box": [int(v) for v in b]}
        for t, s, b in zip(res["rec_texts"], res["rec_scores"], boxes)
    ]
    return {"lines": lines, "seconds": elapsed}


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #
def _find_gutter(lines, width):
    """Return the x position of a vertical gutter separating two columns, or None.

    A gutter is an x position in the central 15-85% of the page that (almost) no
    box crosses, while there are boxes on both sides that cover a large share of
    the page height. Boxes that cross it (full-width headers) are allowed only if
    they are few.
    """
    if len(lines) < 8 or width <= 0:
        return None
    boxes = np.array([l["box"] for l in lines])
    cover = np.zeros(width + 1, dtype=int)
    for x0, _, x1, _ in boxes:
        cover[max(x0, 0):min(x1, width) + 1] += 1
    lo, hi = int(width * 0.15), int(width * 0.85)
    candidates = np.where(cover[lo:hi] <= max(1, int(0.05 * len(lines))))[0] + lo
    if len(candidates) == 0:
        return None
    # take the widest run of low-coverage x positions
    runs = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
    run = max(runs, key=len)
    if len(run) < width * 0.01:
        return None
    gx = int(run.mean())
    left = boxes[boxes[:, 2] <= gx]
    right = boxes[boxes[:, 0] >= gx]
    if len(left) < 4 or len(right) < 4:
        return None
    page_h = boxes[:, 3].max() - boxes[:, 1].min()
    span = lambda b: (b[:, 3].max() - b[:, 1].min()) / max(page_h, 1)
    if span(left) < 0.4 or span(right) < 0.4:
        return None
    return gx


def _rows(lines):
    """Group boxes into visual rows (similar y centre) and sort left-to-right."""
    if not lines:
        return []
    items = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    heights = [l["box"][3] - l["box"][1] for l in items]
    tol = 0.5 * float(np.median(heights))
    rows, cur, cur_y = [], [items[0]], (items[0]["box"][1] + items[0]["box"][3]) / 2
    for l in items[1:]:
        yc = (l["box"][1] + l["box"][3]) / 2
        if abs(yc - cur_y) <= tol:
            cur.append(l)
            cur_y = np.mean([(c["box"][1] + c["box"][3]) / 2 for c in cur])
        else:
            rows.append(sorted(cur, key=lambda c: c["box"][0]))
            cur, cur_y = [l], yc
    rows.append(sorted(cur, key=lambda c: c["box"][0]))
    return rows


def reading_order(lines, width=None):
    """Return lines (dicts) in reading order plus the detected layout.

    1. Single column: group boxes into rows by their vertical centre (tolerance =
       half the median line height) and read rows top-to-bottom, boxes inside a
       row left-to-right, so "Staff Nurse ... 2014 - 2019" stays on one line.
    2. Two columns: if a vertical gutter exists (see `_find_gutter`), boxes that
       cross it (e.g. the name banner) are read first if they sit above the
       columns, then the whole left column, then the whole right column. Reading
       column by column keeps each job title next to its own dates instead of
       interleaving the sidebar with the experience section.
    """
    if not lines:
        return [], "empty"
    width = width or max(l["box"][2] for l in lines)
    gx = _find_gutter(lines, width)
    if gx is None:
        return [l for row in _rows(lines) for l in row], "single"
    left = [l for l in lines if l["box"][2] <= gx]
    right = [l for l in lines if l["box"][0] >= gx]
    spanning = [l for l in lines if l["box"][0] < gx < l["box"][2]]
    col_top = min(l["box"][1] for l in left + right)
    head = [l for l in spanning if l["box"][3] <= col_top + 5]
    tail = [l for l in spanning if l not in head]
    order = []
    for group in (head, left, right, tail):
        order += [l for row in _rows(group) for l in row]
    return order, "two-column"


def ordered_text(ocr_json: dict, min_score: float = 0.0) -> str:
    lines = [l for l in ocr_json["lines"] if l["score"] >= min_score]
    order, _ = reading_order(lines, ocr_json.get("width"))
    rows, last_y = [], None
    for l in order:
        yc = (l["box"][1] + l["box"][3]) / 2
        h = l["box"][3] - l["box"][1]
        if rows and last_y is not None and abs(yc - last_y) <= 0.5 * h:
            rows[-1] += " " + l["text"]  # same visual row -> same text line
        else:
            rows.append(l["text"])
        last_y = yc
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
def process_all(limit=None, shard=0, num_shards=1, cpu_threads=None):
    from PIL import Image

    ensure_dirs()
    ids_file = RAW / "sampled_ids.csv"
    if ids_file.exists():
        ids = pd.read_csv(ids_file, dtype={"id": str})["id"].tolist()
    else:  # download still running: process whatever images already exist
        ids = sorted(p.stem for p in RAW_IMAGES.glob("*.png"))
    ids = [i for k, i in enumerate(ids) if k % num_shards == shard]
    todo = [i for i in ids if not (OCR_DIR / f"{i}.json").exists()]
    if limit:
        todo = todo[:limit]
    print(f"[shard {shard}] {len(todo)} images to process", flush=True)
    if not todo:
        return
    ocr = build_ocr(cpu_threads)
    t_start = time.time()
    for k, cv_id in enumerate(todo, 1):
        path = RAW_IMAGES / f"{cv_id}.png"
        try:
            w, h = Image.open(path).size
            out = run_ocr(ocr, str(path))
            out.update(id=cv_id, width=w, height=h, ok=True, error=None)
        except Exception as e:  # keep going, record the failure
            out = dict(id=cv_id, lines=[], seconds=None, ok=False, error=repr(e))
        out.update(det_model=DET_MODEL, rec_model=REC_MODEL)
        save_json(out, OCR_DIR / f"{cv_id}.json")
        if k % 25 == 0:
            rate = (time.time() - t_start) / k
            print(f"[shard {shard}] {k}/{len(todo)}  {rate:.2f}s/CV  "
                  f"ETA {rate * (len(todo) - k) / 60:.0f} min", flush=True)


def summary():
    rows = []
    for p in OCR_DIR.glob("*.json"):
        j = load_json(p)
        scores = [l["score"] for l in j["lines"]]
        layout = reading_order(j["lines"], j.get("width"))[1] if j["lines"] else "empty"
        rows.append(dict(id=j["id"], ok=j["ok"], seconds=j["seconds"], n_lines=len(scores),
                         mean_conf=np.mean(scores) if scores else np.nan,
                         min_conf=np.min(scores) if scores else np.nan,
                         low_conf_share=np.mean(np.array(scores) < 0.8) if scores else np.nan,
                         layout=layout, error=j.get("error")))
    df = pd.DataFrame(rows)
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})[["id", "occupation"]]
    df = df.merge(meta, on="id", how="left")
    df.to_csv(TABLES / "ocr_per_cv.csv", index=False)
    from importlib.metadata import version  # importing paddleocr here would pull torch in after paddle (DLL clash on Windows)

    stats = dict(
        n_cvs=len(df), n_failed=int((~df["ok"]).sum() + (df["n_lines"] == 0).sum()),
        failed_ids=df.loc[~df["ok"] | (df["n_lines"] == 0), "id"].tolist(),
        seconds_per_cv_mean=df["seconds"].mean(), seconds_per_cv_median=df["seconds"].median(),
        total_hours=df["seconds"].sum() / 3600,
        mean_confidence=df["mean_conf"].mean(), mean_lines_per_cv=df["n_lines"].mean(),
        share_lines_below_0_8=df["low_conf_share"].mean(),
        layout_counts=df["layout"].value_counts().to_dict(),
        det_model=DET_MODEL, rec_model=REC_MODEL,
        paddleocr=version("paddleocr"), paddlepaddle=version("paddlepaddle"),
        hardware=f"{platform.processor()} | {os.cpu_count()} logical cores | CPU only",
        by_occupation=df.groupby("occupation")[["seconds", "mean_conf", "n_lines"]].mean().to_dict(),
    )
    save_json(stats, METRICS / "ocr_stats.json")
    print({k: v for k, v in stats.items() if k != "failed_ids"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--cpu-threads", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        process_all(a.limit, a.shard, a.num_shards, a.cpu_threads)
