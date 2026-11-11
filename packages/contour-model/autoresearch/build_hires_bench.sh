#!/bin/bash
# High-resolution robustness benchmark: 2560-4096 px renders of held-out test/val designs.
cd "$(dirname "$0")/.."
export PYTHONPATH=src:../../research/synthetic-shapes/src
nice -n 12 .venv/bin/python -u -m contour_model.extract_corpus "/media/kyle/T7 Shield/inspection-contour-training/primitives-intent-v1-20260920" "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-bench-hires" --splits test --limit 120 --sides 2560,3200,4096 --max-tokens 32768 --workers 3 --seed 4242
