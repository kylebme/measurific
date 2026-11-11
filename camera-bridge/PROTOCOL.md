# ICAM protocol v1

The protocol is private to the app but deliberately independent from Electron
and Python.

stdin commands are UTF-8 lines:

```text
discover<TAB>request-id
open<TAB>request-id<TAB>percent-encoded-device-id
start<TAB>request-id
pause<TAB>request-id
meter<TAB>request-id<TAB>rectangle<TAB>x0<TAB>y0<TAB>x1<TAB>y1
meter<TAB>request-id<TAB>circle<TAB>cx<TAB>cy<TAB>r<TAB>0
focus<TAB>request-id<TAB>rectangle<TAB>x0<TAB>y0<TAB>x1<TAB>y1
focus<TAB>request-id<TAB>circle<TAB>cx<TAB>cy<TAB>r<TAB>0
set<TAB>request-id<TAB>percent-encoded-name<TAB>percent-encoded-value
quit<TAB>request-id
```

Every stdout packet starts with a 16-byte little-endian prefix:

| Offset | Size | Meaning |
| --- | ---: | --- |
| 0 | 4 | ASCII `ICAM` |
| 4 | 1 | protocol version (`1`) |
| 5 | 1 | kind: JSON `1`, preview `2`, snapshot `3` |
| 6 | 2 | reserved |
| 8 | 4 | JSON header byte count |
| 12 | 4 | binary payload byte count |

`set` accepts `exposure`, `exposureAuto`, `gain`, `gainAuto` and `channel`.
`channel` selects the delivered image:
- `camera`: the camera's own grayscale output. On a color camera this is
  demosaiced and color-processed in the camera.
- `red`, `green` or `blue`: one Bayer color site from the camera's deepest
  unpacked Bayer format, at half resolution and never demosaiced. Green uses the
  green site in red's row, so every channel lies on a regular 2-pixel grid.

For single channels, optional color processing that would alter raw samples is
turned off: automatic white balance, the light-source preset, the color space,
LUT and gamma, with balance ratios set to 1. The camera's values are saved first
and restored for `camera` output and on close. The response to `open`, `start` and
`set channel` includes the camera's current `channel`, the `channels` it supports
(`["camera"]` without raw Bayer output), the delivered `width`/`height` and the
`sensorWidth`/`sensorHeight`. A camera opens with `camera` output.

The UTF-8 JSON header immediately follows the prefix, then the optional image
payload. Image payloads are tightly packed grayscale rows. Previews are 8-bit.
Snapshots use 8-bit storage for 8-bit sources and little-endian 16-bit storage
for higher-depth sources; `sensorBits` records the meaningful source depth.

Frame headers give `fullWidth`/`fullHeight` (the delivered frame), `channel`,
`sampleStep` and `sampleOffsetX`/`sampleOffsetY`. Delivered pixel (x, y) is sensor
pixel (sampleStep·x + sampleOffsetX, sampleStep·y + sampleOffsetY).

`meter` reads every delivered (native-resolution, or single-channel) sample whose
center is inside the normalized ROI. Circle radius is relative to the shorter image dimension.
It returns `count`, `clipped` (sensor maximum), `peak` and `percentile` (99.9th),
with intensities normalized to the sensor maximum, plus frame `sequence`.
No preview decimation or synthetic ROI fill contributes to these statistics.

`focus` measures native delivered 5x5 neighborhoods wholly inside the ROI,
with at most 65536 centers. It returns `score` (relative sharpness), `count`,
`status` (`ok`, `too-small`, `low-detail`, or `unavailable`) and frame `sequence`.
Only `ok` scores are usable. No preview scaling, bit-depth reduction, or ROI
mask fill contributes. The app requests it only while streaming, at up to 5 Hz.
See [algorithm and validation](../app/docs/live-focus.md).
