"""Macro-scale multi-part backlit scenes for coarse part/edge segmentation."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
for path in (os.path.join(REPO, "synthetic-shapes", "src"), os.path.join(REPO, "synthetic-renders"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

PROFILE = "macro_v1"
SIZE = 512            # macro image side, 1/4 of the 2048 source renders
FULL_SCALE = 4        # full-resolution pixels per macro pixel
