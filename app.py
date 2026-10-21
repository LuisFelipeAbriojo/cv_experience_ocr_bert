"""Streamlit app - Predicting years of experience from CV images (PaddleOCR + BERT).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import torch  # noqa: F401  must be imported before paddle (Windows DLL clash otherwise)
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
TABLES, FIG, METRICS = ROOT / "output" / "tables", ROOT / "output" / "figures", ROOT / "output" / "metrics"
ANSWERS = ROOT / "output" / "analysis_answers.md"

st.set_page_config(page_title="CV experience estimator", layout="wide")


# --------------------------------------------------------------------------- helpers
def table(name):
    p = TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def jload(name):
    p = METRICS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fig(name, caption=None):
    p = FIG / f"{name}.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        st.info(f"Figure `{name}.png` not generated yet - run `python -m src.visualization`.")


def answer(section: str) -> str:
    """Return one '## <section>' block from output/analysis_answers.md."""
    if not ANSWERS.exists():
        return "_Run the pipeline to generate the analysis._"
    blocks = re.split(r"(?m)^## ", ANSWERS.read_text(encoding="utf-8"))
    for b in blocks:
        if b.startswith(section):
            return b.split("\n", 1)[1]
    return ""


@st.cache_data
def cvs():
    p = ROOT / "data" / "processed" / "cvs.csv"
    return pd.read_csv(p, dtype={"id": str}) if p.exists() else None


@st.cache_resource(show_spinner="Loading PaddleOCR ...")
def get_ocr():
    from src.ocr_extraction import build_ocr
    return build_ocr()


@st.cache_resource(show_spinner="Loading models ...")
def get_models():
    from src.experiments import load_models
    return load_models()


ocr_stats, ocr_q, pre = jload("ocr_stats"), jload("ocr_quality"), jload("preprocessing_stats")
cmp_table = table("model_comparison")

st.title("Predicting years of experience from CV images")
st.caption("PaddleOCR → text cleaning → Median · TF-IDF + Ridge · Embeddings + Ridge · fine-tuned BERT "
           "(+ a date-counting rule as bonus) · PopResume, 4,000 synthetic CVs")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Problem & Methodology", "2 · OCR & Exploratory Analysis",
                                  "3 · Model Results", "4 · Interactive Estimator"])

# =========================================================================== TAB 1
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.header("Objective")
        st.markdown("""
Read a CV **image**, extract its text with **PaddleOCR**, and predict the candidate's **years of
work experience (0-20)**. The number is almost never written on the CV: a model must find every
job, read its start and end dates, add the durations (without double-counting overlaps) and ignore
the education dates. A recruiter does this in their head; here four models of increasing
complexity try to learn it from text.
""")
        st.header("Data and target")
        st.markdown(f"""
* **PopResume** (Yu, Park & Moon, EMNLP 2026, CC BY 4.0): 60,884 synthetic resumes, 5 occupations,
  generated from U.S. population statistics - no real person's data.
* **Sample:** 800 **no-photo** images per occupation (4,000 total), streamed per occupation,
  `filter(has_photo == False)` → `shuffle(seed=42, buffer_size=5000)` → `take(800)`; IDs in
  `data/raw/sampled_ids.csv`.
* **Target:** `exp_year` clipped to 0-20 (values above 20 mean "20 or more").
  {pre.get('n_at_cap', '?')} sampled CVs sit at the cap ({100 * pre.get('share_at_cap', 0):.1f}%), far
  below the 15% threshold, so the cap stays at 20 for every model.
* **Never used as features:** age, sex, race, names, state/region. The reference `text` column is
  used **only** to measure OCR error (and for the OCR-noise experiment), never to train.
* **Masking rule:** phrases stating total experience, age or date of birth are replaced by
  `[MASKED]`. CVs affected: **{pre.get('n_cvs_masked', '?')}** (job and graduation dates are kept -
  they are the evidence).
""")
        st.header("OCR setup and reading order")
        st.markdown(f"""
* PaddleOCR {ocr_stats.get('paddleocr', '3.x')} (PaddlePaddle {ocr_stats.get('paddlepaddle', '')}),
  detection **{ocr_stats.get('det_model', 'PP-OCRv5_mobile_det')}**, recognition
  **{ocr_stats.get('rec_model', 'en_PP-OCRv5_mobile_rec')}**; no orientation / unwarping
  (clean digital renders). Mobile models were chosen because everything runs on a laptop CPU.
* Every line is cached with text, confidence and box (`data/ocr/<id>.json`).
* **Reading order:** boxes are grouped into visual rows (same vertical centre ± half a line
  height) and read top-to-bottom, left-to-right, so a job title and its dates stay on one line.
  A **gutter detector** looks for a vertical strip that no box crosses; if found, full-width
  headers are read first, then the whole left column, then the right one.
""")
    with c2:
        st.header("Pipeline")
        st.markdown("""
```
CV image (PNG)
   │ PaddleOCR (det + rec)  → lines, scores, boxes
   ▼
reading order → cleaning → PII removal → masking
   ▼
split 70/15/15 (stratified by occupation × band, seed 42)
   ▼
┌─────────┬──────────────┬────────────────┬──────────────┐
│ Median  │ TF-IDF+Ridge │ bge-small+Ridge│ DistilBERT   │
│         │              │ (frozen)       │ fine-tuned   │
└─────────┴──────────────┴────────────────┴──────────────┘
   ▼
same test set → MAE, RMSE, ±k acc, bands, R², bootstrap CIs
```
""")
        st.header("The models")
        st.markdown("""
| Model | What it can use |
|---|---|
| **Median** | nothing - always the training median |
| **TF-IDF + Ridge** | which words / years appear, not how they combine |
| **Embeddings + Ridge** | the overall "meaning" of the CV, frozen encoder |
| **Fine-tuned BERT** | full token sequence, can learn to relate dates |
| **Date rules** (bonus) | regex date ranges, merged and added up |
""")
        st.header("Limitations")
        st.markdown("""
Synthetic resumes with one clean template; only 5 occupations; U.S. statistics; English only;
CPU-only training (small model, few epochs). The estimate is a **support tool**: it must never
decide who is hired.
""")

# =========================================================================== TAB 2
with tab2:
    df = cvs()
    st.header("Years of experience")
    fig("target_distribution")
    st.caption("Histogram + boxplot chosen to show both the overall skew and the differences between "
               "occupations. Most candidates are junior (median ~5 years) and very few reach the cap, "
               "so errors on seniors weigh little in the average but matter a lot to recruiters.")
    if df is not None:
        c = st.columns(4)
        c[0].metric("CVs", len(df))
        c[1].metric("Median years", f"{df.y.median():.1f}")
        c[2].metric("At cap (20)", int((df.exp_year >= 20).sum()))
        c[3].metric("CVs masked", pre.get("n_cvs_masked", "?"))

    st.header("OCR quality")
    if ocr_q:
        c = st.columns(5)
        c[0].metric("Mean CER", f"{100 * ocr_q['cer_mean']:.2f}%")
        c[1].metric("Mean WER", f"{100 * ocr_q['wer_mean']:.2f}%")
        c[2].metric("Years read correctly", f"{100 * ocr_q['year_recall']:.2f}%")
        c[3].metric("Date ranges read correctly", f"{100 * ocr_q['range_recall']:.2f}%")
        c[4].metric("Mean confidence", f"{ocr_stats.get('mean_confidence', float('nan')):.3f}")
    fig("ocr_quality")
    st.caption("CER/WER histograms show whether errors are spread evenly or concentrated in a few bad "
               "pages; the bar chart isolates what matters most for this task - whether the years were "
               "read correctly, since one wrong digit changes the answer.")
    t = table("ocr_quality_by_occupation")
    if t is not None:
        st.dataframe(t.round(4), use_container_width=True, hide_index=True)
    if ocr_stats:
        st.markdown(f"**OCR run:** {ocr_stats.get('seconds_per_cv_mean', 0):.2f} s/CV on "
                    f"{ocr_stats.get('hardware', '')}; failed images: {ocr_stats.get('n_failed', 0)}; "
                    f"layouts: {ocr_stats.get('layout_counts', {})}.")

    st.header("Example: PaddleOCR boxes and extracted text")
    if df is not None:
        ex_id = st.selectbox("CV id", df.id.head(200).tolist(), key="ex_id")
        a, b = st.columns(2)
        img_p = ROOT / "data" / "raw" / "images" / f"{ex_id}.png"
        ocr_p = ROOT / "data" / "ocr" / f"{ex_id}.json"
        with a:
            if img_p.exists() and ocr_p.exists():
                from PIL import Image, ImageDraw
                im = Image.open(img_p).convert("RGB")
                d = ImageDraw.Draw(im)
                for l in json.loads(ocr_p.read_text(encoding="utf-8"))["lines"]:
                    d.rectangle(l["box"], outline=(59, 111, 182) if l["score"] >= .8 else (214, 69, 65), width=2)
                st.image(im, caption="Blue: confidence ≥ 0.8, red: < 0.8", use_container_width=True)
            else:
                fig("ocr_example_boxes", "Images are not in the repo - regenerate with src/download_data.py")
        with b:
            row = df[df.id == ex_id].iloc[0]
            st.markdown(f"**True years:** {row.y:.2f}  ·  occupation: {row.occupation}")
            st.text_area("Cleaned OCR text (model input)", row.text, height=420,
                         key=f"ex_txt_{ex_id}")  # key per CV so the box refreshes
    st.caption("Shown side by side so the reader can check reading order: each job title must stay "
               "on the same line as its dates.")

# =========================================================================== TAB 3
with tab3:
    st.header("Model comparison (test set)")
    if cmp_table is not None:
        st.dataframe(cmp_table.round(3), use_container_width=True, hide_index=True)
    st.subheader("BERT's MAE gain over each model (95% paired bootstrap CI)")
    g = table("bert_gain_bootstrap")
    if g is not None:
        st.dataframe(g.round(3), use_container_width=True, hide_index=True)
    fig("bert_gain")

    st.header("Predicted vs true")
    opts = {p.stem.replace("pred_vs_true_", ""): p for p in FIG.glob("pred_vs_true_*.png") if "all" not in p.stem}
    if opts:
        k = st.selectbox("Model", list(opts), index=len(opts) - 1)
        c1, c2 = st.columns([1, 1])
        c1.image(str(opts[k]), use_container_width=True)
        c2.markdown("The dashed line is a perfect prediction. Points above it are over-estimates, "
                    "below it under-estimates. A cloud that is flatter than the diagonal means the model "
                    "pulls predictions toward the middle (juniors over-estimated, seniors under-estimated).")
        t = table("mean_error_by_band")
        if t is not None:
            c2.markdown("**Mean signed error (pred − true) by band**")
            c2.dataframe(t.round(2), use_container_width=True, hide_index=True)
    fig("pred_vs_true_all")

    st.header("MAE by experience band and by occupation")
    fig("mae_by_band")
    fig("mae_by_occupation")
    st.header("Band confusion matrices")
    fig("confusion_bands", "Rows: true band, columns: predicted band; colour = row-normalised share.")

    st.header("Largest BERT errors")
    t = table("bert_top10_errors_annotated") if (TABLES / "bert_top10_errors_annotated.csv").exists() \
        else table("bert_top10_errors")
    if t is not None:
        st.dataframe(t, use_container_width=True, hide_index=True)

    st.header("Q1 · What does BERT learn that the baselines cannot?")
    st.markdown(answer("Q1"))
    t = table("probe_q1")
    if t is not None:
        st.dataframe(t.round(3), use_container_width=True, hide_index=True)
    st.header("Q2 · How much does OCR noise cost?")
    t = table("ocr_noise_q2")
    if t is not None:
        st.dataframe(t.round(3), use_container_width=True, hide_index=True)
    st.markdown(answer("Q2"))
    st.header("Q3 · Sensitivity experiment")
    for n in ("sensitivity_cleaning_q3", "sensitivity_bert_q3"):
        t = table(n)
        if t is not None:
            if n == "sensitivity_cleaning_q3":
                t = t.pivot_table(index="model", columns="setup", values="MAE").reset_index()
            st.dataframe(t.round(3), use_container_width=True, hide_index=True)
    st.markdown(answer("Q3"))
    st.header("Q4 · Fairness check and business value")
    t = table("fairness_bert")
    if t is not None:
        st.dataframe(t.round(3), use_container_width=True, hide_index=True)
    st.markdown(answer("Q4"))

# =========================================================================== TAB 4
with tab4:
    st.header("Estimate experience from a CV")
    up = st.file_uploader("Upload a CV image (PNG/JPG) or a PDF (first page is used)",
                          type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        from PIL import Image
        if up.name.lower().endswith(".pdf"):
            import fitz
            doc = fitz.open(stream=up.read(), filetype="pdf")
            pix = doc[0].get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        else:
            img = Image.open(up).convert("RGB")
        from src.ocr_extraction import ordered_text, run_ocr
        from src.preprocessing import clean
        with st.spinner("Running PaddleOCR ..."):
            out = run_ocr(get_ocr(), np.array(img)[:, :, ::-1])
            out["width"] = img.width
            raw = ordered_text(out)
            text, hits = clean(raw)
        c1, c2 = st.columns([1, 1])
        c1.image(img, use_container_width=True)
        c2.caption(f"OCR: {len(out['lines'])} lines in {out['seconds']:.1f}s · masked phrases: {sum(hits.values())}")
        c2.text_area("Extracted and cleaned text", text, height=380,
                     key=f"up_txt_{up.name}_{up.size}")
        with st.spinner("Predicting ..."):
            preds = {name: float(f([text])[0]) for name, f in get_models().items()}
        st.subheader("Predicted years of experience")
        cols = st.columns(len(preds))
        for c, (name, v) in zip(cols, preds.items()):
            c.metric(name, f"{v:.1f} y")

    st.divider()
    st.header("Browse test CVs")
    df = cvs()
    preds_p = {k: METRICS / f"preds_{k}.csv" for k in ("median", "tfidf", "embeddings", "rules", "bert")}
    if df is not None and all(p.exists() for p in preds_p.values()):
        te = pd.read_csv(preds_p["bert"], dtype={"id": str}).rename(columns={"y_pred": "BERT"})
        for k, lab in (("median", "Median"), ("tfidf", "TF-IDF"), ("embeddings", "Embeddings"), ("rules", "Rules")):
            te = te.merge(pd.read_csv(preds_p[k], dtype={"id": str})[["id", "y_pred"]].rename(columns={"y_pred": lab}), on="id")
        te = te.merge(df[["id", "occupation", "text"]], on="id")
        te["BERT abs error"] = (te.BERT - te.y_true).abs()
        c1, c2, c3 = st.columns(3)
        occ = c1.multiselect("Occupation", sorted(te.occupation.unique()), default=sorted(te.occupation.unique()))
        yr = c2.slider("True years", 0.0, 20.0, (0.0, 20.0))
        er = c3.slider("BERT absolute error", 0.0, float(np.ceil(te["BERT abs error"].max())),
                       (0.0, float(np.ceil(te["BERT abs error"].max()))))
        sel = te[te.occupation.isin(occ) & te.y_true.between(*yr) & te["BERT abs error"].between(*er)]
        st.caption(f"{len(sel)} test CVs")
        show = ["id", "occupation", "y_true", "Median", "TF-IDF", "Embeddings", "Rules", "BERT", "BERT abs error"]
        st.dataframe(sel[show].round(2), use_container_width=True, hide_index=True)
        pick = st.selectbox("Open a CV", sel.id.tolist()) if len(sel) else None
        if pick:
            r = sel[sel.id == pick].iloc[0]
            a, b = st.columns(2)
            img_p = ROOT / "data" / "raw" / "images" / f"{pick}.png"
            if img_p.exists():
                a.image(str(img_p), use_container_width=True)
            b.text_area("Cleaned OCR text", r.text, height=420, key=f"browse_txt_{pick}")

        st.header("Good and bad predictions")
        st.markdown(answer("Examples"))
    else:
        st.info("Run the full pipeline to browse test predictions.")
