# Provenance and attribution

The detector modules in `src/pae/` are a maintained derivative of
[`gravi-toni/subpixel-edges`](https://github.com/gravi-toni/subpixel-edges),
specifically commit `a4f8f1a4160f5405d85036b68c7a427568b6cd9d` (5 July
2020). That implementation is Copyright (c) 2020 Giuseppe Vitucci and Nicola
Vitucci and is distributed under the MIT License reproduced in `LICENSE`.

This fork retains the PAE formulation and the ideal-image mode's candidate
ordering. Maintained changes include:

- replacing removed NumPy aliases with current Python/NumPy types;
- adding input validation, type hints, packaging, documentation, and tests;
- making every public output feature a one-dimensional array;
- defining the zero-curvature iterative reconstruction as its documented
  infinite-radius limit (the upstream Python variable was uninitialized);
- using column-major working arrays and deriving candidate coordinates from
  linear indices to remove avoidable full-frame copies; MATLAB parity tests
  establish equivalence.
- adding an OpenCV threshold-contour seeding path for controlled silhouettes;
  it applies the same candidate predicates and PAE equations directly to the
  original grayscale samples and is tested for bit-exact parity with the full
  unsmoothed detector.
- accepting external seed contours and exposing the compensated noise modes
  through the seeded API;
- correcting a transposed close-edge replacement index in the low-noise path,
  and using derivatives rather than intensities for iterative close-edge tests;
- rejecting invalid/zero-contrast fits without misaligning result arrays;
- replacing the asymmetric sampled-circle reconstruction with analytic
  disk/pixel intersection areas, including tangent crescents. This intentionally
  changes iterative outputs; independent area and localization tests supplement
  the historical MATLAB fixtures. See `INVESTIGATION.md` for numerical evidence.

The upstream port is itself an implementation of:

> A. Trujillo-Pino, K. Krissian, M. Alemán-Flores, D. Santana-Cedrés,
> “Accurate Subpixel Edge Location Based on Partial Area Effect,” *Image and
> Vision Computing* 31(1), 72–90 (2013),
> https://doi.org/10.1016/j.imavis.2012.10.005

The author-maintained MATLAB reference is “Accurate subpixel edge location,”
version 2.13.4, by Agustin Trujillo-Pino:
https://www.mathworks.com/matlabcentral/fileexchange/48908

No MATLAB source code is included in this package. “MATLAB” is a trademark of
The MathWorks, Inc.; its mention here is solely for implementation provenance
and numerical validation.
