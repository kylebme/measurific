"""Subpixel mask super-resolution: 48x48 backlit patch -> 16x binary mask."""

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.environ.get(
    "EDGESR_SOURCE",
    "/media/kyle/T7 Shield/inspection-synthetic-renders/full-catalog-5gb-2048")
WORK_DIR = os.environ.get("EDGESR_WORK", os.path.join(REPO_DIR, "work"))
CACHE_DIR = os.path.join(WORK_DIR, "cache")
RUNS_DIR = os.path.join(WORK_DIR, "runs")

SCALE = 16
