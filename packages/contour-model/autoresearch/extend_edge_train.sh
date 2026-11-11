#!/bin/bash
# Extend the 4,000-design edge train corpus to every train design in a new directory
# (the 4,000-design corpus stays frozen for running experiments). The copy carries its
# receipts, so the prefix is not re-extracted (selection is prefix-consistent).
cd "$(dirname "$0")/.."
export PYTHONPATH=src:../../research/synthetic-shapes/src
[ -d "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train-full" ] || { cp -r "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train" "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train-full.tmp" && rm -f "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train-full.tmp/audit.json" && mv "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train-full.tmp" "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train-full"; }
nice -n 12 .venv/bin/python -u -m contour_model.extract_corpus "/media/kyle/T7 Shield/inspection-contour-training/primitives-intent-v1-20260920" "/media/kyle/T7 Shield/inspection-contour-training/primitives-edge-v1-train-full" --splits train --workers 6
