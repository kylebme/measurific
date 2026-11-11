#!/bin/bash
# Resumable: rerun to continue (receipts). Benchmark (test+val) first, then training renders.
# Train selection is prefix-consistent: raising --limit later extends the same corpus.
cd "$(dirname "$0")/.."
export PYTHONPATH=src:../../research/synthetic-shapes/src
until [ -f "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-bench/audit.json" ]; do
  pgrep -f "[p]rimitives-edge-v1-bench" >/dev/null || nice -n 12 .venv/bin/python -u -m contour_model.extract_corpus "/media/kyle/T7 Shield/inspection-contour-training/primitives-intent-v1-20260920" "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-bench" --splits test,val --workers 12 --save-png
  sleep 30
done
nice -n 12 .venv/bin/python -u -m contour_model.extract_corpus "/media/kyle/T7 Shield/inspection-contour-training/primitives-intent-v1-20260920" "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train" --splits train --workers 11 --limit ${TRAIN_LIMIT:-4000}
