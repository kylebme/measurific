# Live sharpness indicator: algorithm decision

Research and local comparison completed before feature implementation (2026-10-01).

## Decision

Use **lightly smoothed, noise-corrected Tenengrad, normalized by signal variance**
on native delivered camera pixels inside the current rectangle/circle ROI. This is
an engineering adaptation of Tenengrad, not a published universally best operator.
Show a relative numeric sharpness score and the best score seen during this live
session. Higher means sharper for the same stationary scene/ROI. A session peak is
not proof of correct optical focus. The indicator belongs only to the live camera
panel; paused captures and opened images do not measure or display it.

## Evidence and alternatives

- [Pertuz, Puig & Garcia (2013)](https://doi.org/10.1016/j.patcog.2012.11.011)
  compared 36 operators for shape-from-focus. Laplacian operators performed best
  under ordinary conditions, but were particularly noise-sensitive. Those are
  small-window depth reconstruction results, not a direct ranking for this GUI.
  The [authors' supplementary page](https://sites.google.com/view/cvia/focus-measure)
  explicitly says ranking depends on the camera and scene.
- [Piao et al. (2025)](https://doi.org/10.3390/s25103144) evaluates microscopy focus
  curves and reports Tenengrad's favorable sensitivity/noise balance. This supports
  a first derivative for live manual focus, without proving camera-specific quality.
- **Variance of Laplacian / modified Laplacian:** sensitive near focus, inexpensive,
  but noise can dominate the mostly flat background of silhouette inspection.
- **Brenner:** inexpensive, but the usual one-axis version favors one edge
  direction. Both-axis differences still admit more noise than smoothed Sobel.
- **Image variance / entropy:** inexpensive but describe scene contrast/content as
  much as focus. Global variance barely changes as a broad silhouette edge blurs.
- **FFT/DCT/wavelets:** useful frequency selection but add implementation and
  parameter complexity without a demonstrated advantage on these images.
- **Edge width / MTF50:** physically meaningful for a suitable isolated edge or
  calibration target. Arbitrary curved contours, overlapping edges, threshold
  selection, and changing ROI make a general live indicator harder. No physical
  resolution or measurement-quality claim should be inferred from this score.
- **Learned defocus prediction:** needs representative camera/lens training and
  calibration, with additional inference cost. The existing learned contour/edge
  pipeline is inappropriate for an immediate live feedback loop.

## Reproducible pre-implementation experiment

Run `app/.venv/bin/python app/scripts/benchmark-focus.py`. Exact per-image curves
are in `focus-benchmark.json`. Four native-resolution demo silhouettes, six
Gaussian blur sigmas (0, 0.5, 1, 2, 4, 8 delivered pixels), and additive Gaussian
noise (0, 1%, 4% of full scale), clipped to the legal range; fixed RNG seed 42.
These are synthetic perturbations of synthetic images, not measured lens sweeps.

At 4% noise, all four curves were strictly decreasing for both-axis Brenner,
Tenengrad, smoothed Tenengrad, and its uncorrected normalized variant.
The corrected variant was strictly decreasing on three of the four curves;
the fourth had a 0.37% reversal between sigma=0 and sigma=0.5 under extreme noise.
Neither raw Laplacian candidate was monotonic on any of the four images.
Mean score at sigma=8 relative to sigma=0 was 0.937 for Laplacian variance,
0.647 for raw Tenengrad, 0.420 for smoothed Tenengrad, and 0.179 for the selected
variant. Image variance's ratio was 0.937: insufficient focus sensitivity here.
At sigma=1, the selected variant's ratio was 0.821 versus variance's 0.995.
For unclipped affine intensity scaling `I'=0.5 I+0.1`, normalized variants retained
exactly the same score; unnormalized energy scores fell to one quarter.
Noise correction reduced the mean noise bias at sigma=4:
139% without correction versus -0.6% with correction at the extreme 4% noise.
Clipping and non-white noise violate the correction's assumptions. This limitation
is why the UI resets the reference on acquisition changes and remains relative.

## Formula and implementation constraints

Smooth with separable `[1,2,1]/4`, then normalized Sobel derivatives. The combined
5x5 kernels are `D ⊗ S / 128` and `S ⊗ D / 128`, with
`D=[-1,-2,0,2,1]`, `S=[1,4,6,4,1]`.
Estimate white-noise sigma using median absolute diagonal Haar coefficients
`abs(a-b-c+d)/2 / 0.67448975`. Mostly flat silhouette backgrounds suit this robust
estimate. Fine texture, correlated noise, clipping, and hot pixels can bias it.
The combined derivative kernels have total squared norm `1400/16384`, giving

`score = 1000 max(0, mean(gx²+gy²) - 1400/16384 sigma²) / (var(I)-sigma²)`.

Reject insufficient samples, near-flat regions (signal standard deviation under
0.5% full scale), and noise-dominated regions (signal variance <= 4 sigma²).
No adaptive edge threshold, histogram stretching, per-frame resizing, or changing
sampling grid: those can distort the focus curve. The scale factor only makes
numbers readable; it is not a percentage or universal sharpness threshold.

Measure at most 5 times/sec in the native bridge, with bounded stratified samples
of native 5x5 neighborhoods (up to 65536 centers). Deterministic within-cell offsets
reduce periodic sampling aliasing. All kernel pixels must lie inside the ROI; no
mask-filled boundary contributes. Bayer channels use their native delivered site
grid, before preview decimation or 8-bit quantization. Sampling reduces spatial
coverage, not derivative scale. Very small/sparse features can still be missed;
select an ROI around the relevant edge. Validate native sampling on the same
blur sequences, plus orientation, noise-only, 16-bit, and ROI-boundary cases.

Smooth temporal feedback with a 200 ms exponential time constant, and hold the
peak of that same smoothed score. Reset on resume, ROI, channel, exposure/gain
control changes, and automatic exposure adjustments. Offer a manual reference
reset when repositioning/replacing a part. Mark feedback stale after 1 second
without a new sampled frame. Ignore duplicate sampled frames and never queue multiple focus requests.

## Native implementation validation

Run the same script with `--native` to compile and call the production C metric.
`focus-benchmark-native.json` includes the measured curves and peak locations.
The bounded native metric was monotonic on all four demo scenes without added
noise, and three at both 1% and 4%; the stepped-shaft case had small reversals
between sigma=0 and 0.5 after spatial sampling (see the stored curves). Its mean blur8/sharp ratio was 0.176 and
noise inflation at sigma=4 was -0.5%. This illustrates why individual noisy
samples cannot certify the exact best focal position, and why a stable ROI and
temporal smoothing are useful. Gaussian defocus sigma is not a lens-position unit.

The local native benchmark measured 1.69 ms median and 1.80 ms p95 per
metric call on in-memory grayscale pixels (this machine/build only; excludes
camera transport and conversion). A real `CameraService` run against the bundled
Aravis fake camera also produced focus feedback, reset on ROI change, and
captured/paused successfully. Browser QA checked the live score/bar, peak reset,
low-detail/stale messages, and removal of the panel on pause.

Native C tests cover decreasing analytic edge curves in three orientations,
brightness scaling, 8-/16-bit parity, flat/noise-only rejection, all neighborhood
pixels remaining inside rectangle/circle ROI, and bounded work on large and very
thin images. Electron tests cover smoothing, peak reset, duplicate frames,
late ROI/pause replies, throttling, and nonfatal metric errors.

## Remaining hardware validation

Real stationary focus sweeps are needed on the supported camera/lens/backlight
combinations, with fixed exposure/gain, varied ROI/edge orientations, and ordinary
operating noise. Confirm the peak matches visual/measured optical focus and that
sampling preserves near-focus sensitivity. This initial choice is supported by
literature and local proxies, not certified for every sensor or scene.
