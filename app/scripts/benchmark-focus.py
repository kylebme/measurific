"""Compare focus candidates on repository silhouettes before selecting a live metric.
Run: app/.venv/bin/python app/scripts/benchmark-focus.py
Synthetic Gaussian defocus/noise is a proxy, not camera calibration.
"""
from pathlib import Path
import json
import sys
import ctypes
import subprocess
import tempfile
from time import perf_counter
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve1d, laplace

ROOT = Path(__file__).resolve().parents[1]
BLURS = [0, .5, 1, 2, 4, 8]
SMOOTH = np.array([1, 2, 1]) / 4

def metrics(image):
    gx = convolve1d(convolve1d(image, [-1, 0, 1], axis=1), [1, 2, 1], axis=0) / 8
    gy = convolve1d(convolve1d(image, [-1, 0, 1], axis=0), [1, 2, 1], axis=1) / 8
    b = convolve1d(convolve1d(image, SMOOTH, axis=0), SMOOTH, axis=1)
    bx = convolve1d(convolve1d(b, [-1, 0, 1], axis=1), [1, 2, 1], axis=0) / 8
    by = convolve1d(convolve1d(b, [-1, 0, 1], axis=0), [1, 2, 1], axis=1) / 8
    crop = np.s_[2:-2, 2:-2]
    e = (bx[crop]**2 + by[crop]**2).mean()
    variance = image[crop].var()
    diagonal = (image[:-1,:-1] - image[1:,:-1] - image[:-1,1:] + image[1:,1:]) / 2
    noise2 = (np.median(np.abs(diagonal)) / .67448975)**2
    corrected = max(0, e - (1400/16384) * noise2) / max(variance-noise2, 1e-12)
    return {
        'laplacian_variance': laplace(image)[crop].var(),
        'modified_laplacian': (np.abs(convolve1d(image, [1,-2,1], axis=0))[crop] + np.abs(convolve1d(image, [1,-2,1], axis=1))[crop]).mean(),
        'brenner_xy': ((image[2:, :-2]-image[2:, 2:])**2 + (image[:-2, 2:]-image[2:, 2:])**2).mean(),
        'tenengrad': (gx[crop]**2 + gy[crop]**2).mean(),
        'smoothed_tenengrad': e,
        'variance': variance,
        'normalized_smoothed_tenengrad': e / max(variance, 1e-12),
        'noise_corrected_normalized_tenengrad': corrected,
    }

# --native exercises the production C implementation, including its quantization,
# bounded stratified sampling, ROI interior and detail/noise rejection.
if '--native' in sys.argv:
    class NativeResult(ctypes.Structure):
        _fields_ = [('score', ctypes.c_double), ('count', ctypes.c_uint), ('status', ctypes.c_char_p)]
    native_directory = tempfile.TemporaryDirectory(prefix='inspection-focus-')
    directory = Path(native_directory.name)
    wrapper = directory / 'wrapper.c'
    wrapper.write_text("""
#include "focus.h"
typedef struct { const uint16_t *pixels; int width; } Image;
static uint16_t read_pixel(const void *context, int x, int y) {
    const Image *image = context; return image->pixels[(size_t)y*image->width+x];
}
FocusResult focus_image(const uint16_t *pixels, int width, int height) {
    Image image = {pixels, width}; FocusRoi roi = {0,0,0,1,1};
    return measure_focus(width,height,65535,roi,read_pixel,&image);
}
""")
    library = directory / 'focus.so'
    subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-include', 'stddef.h',
                    '-I', str(ROOT.parent / 'camera-bridge/src'), str(wrapper),
                    str(ROOT.parent / 'camera-bridge/src/focus.c'), '-lm', '-o', str(library)], check=True)
    native = ctypes.CDLL(str(library)).focus_image
    native.restype = NativeResult
    native.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int]
    native_times = []
    def metrics(image):
        pixels = np.ascontiguousarray(np.rint(image*65535), dtype=np.uint16)
        started = perf_counter()
        result = native(pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), image.shape[1], image.shape[0])
        native_times.append((perf_counter()-started)*1000)
        if result.status != b'ok':
            raise AssertionError(f'Native metric rejected demo: {result.status!r}')
        return {'native_selected_metric': result.score}

rng = np.random.default_rng(42)
images = sorted(ROOT.glob('demo-data/*.png'))
rows = []
for path in images:
    source = np.asarray(Image.open(path).convert('L'), dtype=float) / 255
    clean = [metrics(gaussian_filter(source, s) if s else source) for s in BLURS]
    for noise in [0, .01, .04]:
        values = [metrics(np.clip((gaussian_filter(source, s) if s else source) + rng.normal(0, noise, source.shape), 0, 1)) for s in BLURS]
        for name in clean[0]:
            curve = np.array([v[name] for v in values])
            rows.append(dict(image=path.name, metric=name, noise=noise,
                             monotonic=bool(np.all(np.diff(curve)<0)),
                             scores=curve.tolist(), peak_blur=BLURS[int(np.argmax(curve))],
                             blur_1_ratio=curve[2]/curve[0],
                             blur_8_ratio=curve[-1]/curve[0],
                             noise_inflation_at_blur_4=curve[4]/clean[4][name]-1))
print('Mean across four native-resolution demo silhouettes; noise in full-scale intensity units')
for name in clean[0]:
    group = [r for r in rows if r['metric']==name and r['noise']==.04]
    print(f"{name:32} monotonic={sum(r['monotonic'] for r in group)}/{len(group)}  blur1/sharp={np.mean([r['blur_1_ratio'] for r in group]):.3f}  blur8/sharp={np.mean([r['blur_8_ratio'] for r in group]):.3f}  noise inflation at blur4={np.mean([r['noise_inflation_at_blur_4'] for r in group]):.1%}")
print('Affine brightness test (0.5*image + 0.1, no clipping):')
for name in clean[0]:
    print(f"{name:32} ratio={metrics(.5*source+.1)[name]/metrics(source)[name]:.3f}")
if '--native' in sys.argv:
    print(f'Native metric latency: median={np.median(native_times):.2f} ms; p95={np.percentile(native_times,95):.2f} ms (in-memory pixels)')
output = ROOT / ('docs/focus-benchmark-native.json' if '--native' in sys.argv else 'docs/focus-benchmark.json')
output.write_text(json.dumps(dict(blurs=BLURS, seed=42, results=rows), indent=2)+'\n')
