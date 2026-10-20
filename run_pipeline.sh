#!/usr/bin/env bash
# Runs the whole pipeline end to end (each stage is resumable / cached).
# Usage (from the repo root, Git Bash):  bash run_pipeline.sh [--skip-download]
set -euo pipefail
PY=".venv/Scripts/python"
[ -x "$PY" ] || PY="python"
export DISABLE_MODEL_SOURCE_CHECK=True HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

stage() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

if [ "${1:-}" != "--skip-download" ]; then
  stage "0. download (800 no-photo CVs per occupation)"
  for o in registered_nurses elementary_middle_school_teachers software_developers accountants_auditors construction_laborers; do
    $PY -m src.download_data --occupation "$o"
  done
  $PY -m src.download_data
fi
stage "1. PaddleOCR (cached, resumable)";      $PY -m src.ocr_extraction
                                               $PY -m src.ocr_extraction --summary
stage "2a. OCR quality";                       $PY -m src.ocr_quality
stage "2b. cleaning + masking";                $PY -m src.preprocessing
stage "3. split";                              $PY -m src.split
stage "4. baselines";                          $PY -m src.baseline_median
                                               $PY -m src.baseline_tfidf
                                               $PY -m src.baseline_tfidf --no-digits
                                               $PY -m src.baseline_embeddings
                                               $PY -m src.baseline_rules
stage "5. BERT (MSE, cleaned OCR)";            $PY -m src.train_bert --run bert_main
stage "6. evaluation";                         $PY -m src.evaluation
stage "Q1-Q3 experiments";                     $PY -m src.experiments probe
                                               $PY -m src.experiments ocr_noise
                                               $PY -m src.experiments cleaning
stage "Q3 BERT variants";                      $PY -m src.train_bert --run bert_huber --loss huber
                                               $PY -m src.train_bert --run bert_rawocr --text-col text_raw
                                               $PY -m src.experiments collect
stage "figures";                               $PY -m src.visualization
stage "DONE"
