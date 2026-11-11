# SPDX-License-Identifier: MIT
# Derived from gravi-toni/subpixel-edges commit a4f8f1a (2020).
# Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class EdgePixel:
    """
    Class describing pixel features.

    position:
        1D-index inside image
    x, y:
        subpixel position
    nx, ny:
        normal vector (normalized)
    curv:
        curvature
    i0, i1:
        intensities at both sides
    """
    position: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    nx: np.ndarray | None = None
    ny: np.ndarray | None = None
    curv: np.ndarray | None = None
    i0: np.ndarray | None = None
    i1: np.ndarray | None = None

    def __len__(self) -> int:
        return 0 if self.position is None else len(self.position)

    def save(self, filename: str | Path, format: str = "npz") -> None:
        """Save all edge features to a NumPy ``.npz`` archive."""
        if format == "npz":
            np.savez(
                filename,
                position=self.position,
                x=self.x,
                y=self.y,
                nx=self.nx,
                ny=self.ny,
                curv=self.curv,
                i0=self.i0,
                i1=self.i1,
            )
        else:
            raise ValueError(f'File format "{format}" not supported')

    @classmethod
    def load(cls, filename: str | Path, format: str = "npz") -> "EdgePixel":
        """Load an archive, including MATLAB fixtures used for parity tests."""
        if format == "npz":
            with np.load(filename) as edges:
                return cls(**{name: edges[name] for name in _FIELDS})
        if format == "mat":
            from scipy import io

            edges = io.loadmat(filename)
            return cls(**{name: edges[name].ravel("F") for name in _FIELDS})
        else:
            raise ValueError(f'File format "{format}" not supported')


_FIELDS = ("position", "x", "y", "nx", "ny", "curv", "i0", "i1")

# Public name used by the new API. EdgePixel remains for source compatibility.
Edges = EdgePixel
