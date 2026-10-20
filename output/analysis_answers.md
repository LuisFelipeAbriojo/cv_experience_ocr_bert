# Analytical answers

All numbers come from the 600-CV test set (`output/tables/`). MAE is in years.

## Q1

**How much does BERT reduce MAE, and is it significant?**

| Compared with | BERT MAE gain (baseline − BERT) | 95% paired bootstrap CI (2,000 resamples) | Significant |
|---|---|---|---|
| Median (3.51) | **3.30** | [3.08, 3.53] | yes |
| TF-IDF + Ridge (0.87) | **0.67** | [0.61, 0.73] | yes |
| Embeddings + Ridge (1.22) | **1.02** | [0.94, 1.10] | yes |
| Date rules, bonus (0.025) | **−0.18** (BERT is worse) | [−0.20, −0.16] | yes |

Fine-tuned DistilBERT reaches MAE 0.20 years (≈ 2.4 months), 100% of CVs within ±2 years and R² 0.996.
It cuts the TF-IDF error by 77% and the embeddings error by 83%; every interval excludes zero.
However, a 20-line regular expression that adds up the job date ranges is still 8× more accurate
(MAE 0.025) and needs no training: in PopResume, experience *is* the sum of the job durations.

**Where does the gain come from?** BERT's MAE is almost flat across bands (0.23 / 0.18 / 0.19 / 0.24
for 0-2 / 3-5 / 6-10 / 11-20 years), while the baselines degrade at the extremes. The largest absolute
gain over TF-IDF is on **seniors** (1.13 → 0.24, −0.88 years) and juniors (0.87 → 0.23). The
predicted-vs-true plots show the baselines' regression to the middle: on 11-20 year CVs the mean signed
error is −0.91 (embeddings), −0.39 (TF-IDF) and only −0.16 (BERT). The other source of gain is **career
gaps**: on the 242 test CVs with a gap of more than 3 months between jobs, TF-IDF over-estimates by
+0.62 years on average (it reacts to the first-to-last span), whereas BERT's bias stays at −0.01.

**Probe (20 test CVs, 3 edits).** Expected change is what a recruiter would do.

| Edit | Expected | Rules | BERT | TF-IDF | Embeddings |
|---|---|---|---|---|---|
| shift every year by +5 | 0.0 | 0.0 | **−1.90** | −0.67 | −0.66 |
| delete the latest job | −2.65 | −2.61 | −1.46 | −1.45 | −1.68 |
| first job starts 5 y earlier | +4.74 | +4.75 | **+0.49** | +0.60 | +0.47 |

BERT moves in the right direction when a whole job disappears (19/20 CVs), but it does **not** do the
arithmetic: moving one start date 5 years back adds only 0.5 years, and shifting all dates — which
changes nothing for a recruiter — lowers its estimate by 1.9 years. The model reads dates partly as
vocabulary: how recent the years are (every CV ends at 2023.01, so "2028" is out of distribution), how
many job blocks there are and how many tokens the work history takes. Only the rule model behaves like
a recruiter. Fine-tuning buys a large, real improvement over bag-of-words models, but the knowledge it
learns is a statistical shortcut that would break on CVs from another year or template.

## Q2

**OCR quality.** PaddleOCR PP-OCRv5 mobile: mean CER 1.17%, WER 7.9%, mean confidence 0.991, 0 failed
images. **99.996% of the reference years are read with the right digits** (1 miss in 24,383 years) and
99.98% of CVs have every year correct. It fails in two specific ways:

1. it **drops the en dash** in 51% of the job ranges (`2017.10–2020.06` → `2017.102020.06`), which
   breaks any date parser; the cleaning step re-inserts it (range recall 48.6% → 99.6%);
2. on 36 teachers' CVs the job line is so long ("Elementary Middle School Teachers, BrightPath Learning
   Center, 2021.07–2023.01") that the recogniser **loses the last digit** (`2023.0`).

Most of the remaining WER comes from bullets ("•" read as "-" or dropped) and spacing, not from words.

**OCR text vs reference text (same 600 CVs, same models).**

| Model | MAE on OCR | MAE on reference | Gap |
|---|---|---|---|
| TF-IDF + Ridge | 0.874 | 0.880 | −0.005 |
| Embeddings + Ridge | 1.223 | 1.193 | +0.029 |
| Date rules | 0.025 | 0.021 | +0.004 |
| Fine-tuned BERT | 0.203 | 0.242 | −0.040 |

After cleaning, OCR noise costs almost nothing (< 0.04 years for every model). The rule model loses
0.004 years, exactly the truncated teacher months. BERT is even slightly *better* on OCR text: it was
fine-tuned on OCR output, so the reference format (en dashes, "•" bullets, indentation) is a small
domain shift for it. TF-IDF is the most robust (its features are word/char n-grams that are identical
in both texts); embeddings lose the most because a frozen encoder compresses formatting differences
into the same vector as content. **Uncleaned OCR is a different story** — see Q3.

Robustness to a single OCR error is where the rule model is weakest. On CV 59955 (teacher, true 8.06
years) the recogniser cut the last date to `2020.05–2023.` (month lost): the regex cannot parse that
range, drops the whole 2.6-year job and predicts 5.42 — its only large error on the test set. BERT,
which never parses dates explicitly, predicts **7.98** on the same text, and TF-IDF 7.52. Learned models
degrade gracefully; hand-written rules fail hard.

## Q3

**Choice studied: OCR cleaning** (raw OCR in reading order vs cleaned vs cleaned with low-confidence
lines removed) for every model, plus two extra checks (digits in TF-IDF, MSE vs Huber for BERT).

| Model | Raw OCR | Cleaned OCR (main) | Cleaned, lines with conf < 0.98 removed |
|---|---|---|---|
| Median | 3.507 | 3.507 | 3.507 |
| TF-IDF + Ridge | 0.821 | 0.874 | 1.121 |
| Embeddings + Ridge | 1.263 | 1.223 | 1.439 |
| Date rules | **3.138** | **0.025** | 0.593 |
| Fine-tuned BERT | BERT_RAW | 0.203 | — |

* Cleaning is what makes the rule model work: without the dash repair 51% of ranges are unreadable and
  MAE jumps from 0.025 to 3.1 years.
* TF-IDF is slightly better on raw OCR (0.82): the glued token `2017.102020.06` is a single feature
  that carries both dates, so a linear model can use it directly.
* Removing "low-confidence" lines always hurts. Confidence is very high everywhere (no line < 0.95),
  so a 0.98 threshold removes 11.5% of lines, including 6.6% of the job-date lines — the evidence.
  Conclusion: keep every line and repair dates instead of filtering.
* **Digits in TF-IDF:** removing digits raises MAE from 0.87 to **1.69** — TF-IDF depends on year tokens.
* **BERT loss:** MSE 0.203 vs Huber BERT_HUBER (see `sensitivity_bert_q3.csv`).
* **Long-text strategy:** not a real choice here — the longest cleaned CV has ~190 word-pieces, so with
  `max_len=256` head, head+tail and chunking feed identical inputs and every date survives.

## Q4

**Who could use it, and for what.** Recruiting agencies and job boards could **pre-fill** the
"years of experience" field of a profile from an uploaded CV; public employment services could
pre-sort large inflows of CVs for a human counsellor; HR teams could check that a self-declared number
is consistent with the CV. All of these are *assistive* uses where a person sees and can correct the
value. Using the number to **decide who gets an interview** is a different, high-risk task (and an
automated hiring decision under rules such as the EU AI Act or NYC Local Law 144).

**Cost of errors.** ±2 years rarely matters for pre-filling (BERT: 100% within ±2; TF-IDF 93%), but it
matters at hard thresholds ("5+ years required"): a 4.8-year candidate predicted at 5.2 passes, one at
5.2 predicted at 4.8 is silently dropped. ±5 years changes the seniority level (junior ↔ senior) and
would misroute a candidate entirely. **Under-estimation is worse** in screening: the candidate is
filtered out without anyone noticing, while an over-estimate is usually caught at interview. For
salary bands the asymmetry flips (over-estimates cost the employer).

**Fairness check (BERT, test set).**

| Group | n | MAE | Mean error (pred − true) |
|---|---|---|---|
| Female | 307 | 0.188 | −0.002 |
| Male | 293 | 0.218 | −0.001 |
| Asian / Pacific Islander | 106 | 0.182 | −0.019 |
| Black | 31 | 0.247 | −0.126 |
| White | 463 | 0.205 | +0.011 |

There is no systematic gap by sex (bias ≈ 0 for both). By race, Black candidates show a slight
under-estimation (−0.13 years) and higher MAE, but n = 31 is too small to separate this from noise (one
or two large errors move the mean). Sex and race are not inputs and names/contact lines are removed,
so any gap must come through **proxies**: the occupation mix, career length (this group's mean true
experience is the highest, 7.5 years, and BERT under-estimates seniors) and career gaps, which in
PopResume are generated from population statistics that differ by group. A real deployment would need
a much larger audit and a check on the gap distribution per group.

**Limitations.** Synthetic resumes with a single, perfectly regular template (one column, fixed date
format, every CV ending in 2023.01); only 5 occupations; U.S. population statistics; English only; a
clean digital render rather than scanned or photographed CVs; a small model trained on a CPU. The
date-rule result shows how template-specific this is: real CVs use dozens of date formats, overlapping
part-time jobs, "present", and free text. Most importantly, years of experience is a poor proxy for
competence, and the probe shows the model takes shortcuts. Such a system must **never decide who is
hired** — at most it fills a field that a human checks.

## Examples

**Good prediction — CV 05013 (accountant, true 9.87 years).** Three jobs (2011.04–2014.08,
2015.03–2018.06, 2019.10–2023.01) with 1.9 years of gaps. BERT predicts **9.87**; TF-IDF predicts 11.64,
close to the raw first-to-last span (11.8). BERT has learned that gaps do not count.

**Bad prediction — CV 25879 (nurse, true 3.01 years).** Two jobs with a 6-month gap between them. OCR is
perfect (the rule model gives 3.00), but BERT predicts **4.24**: it over-estimates toward the span, the
most common failure among its 10 largest errors (6 of 10 are career gaps; 3 are seniors pulled toward
the mean; none is an OCR, reading-order or truncation problem).
