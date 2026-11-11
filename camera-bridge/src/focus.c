#include "focus.h"
#include <math.h>
#include <stdlib.h>

/* Combined binomial blur and normalized Sobel kernels. Native 5x5 neighborhoods
 * are sampled; pixels are never decimated before taking the derivative. */
FocusResult measure_focus(int width, int height, double maximum, FocusRoi roi,
                          FocusSample sample, const void *context) {
    FocusResult result = {0, 0, "too-small"};
    if (width < 5 || height < 5 || maximum <= 0) return result;
    const double radius = roi.c * fmin(width, height);
    const double left = roi.circle ? roi.a*width-radius : roi.a*width;
    const double right = roi.circle ? roi.a*width+radius : roi.c*width;
    const double top = roi.circle ? roi.b*height-radius : roi.b*height;
    const double bottom = roi.circle ? roi.b*height+radius : roi.d*height;
    const int x0 = (int)fmax(2, ceil(left+1.5));
    const int y0 = (int)fmax(2, ceil(top+1.5));
    const int x1 = (int)fmin(width-2, ceil(right-2.5));
    const int y1 = (int)fmin(height-2, ceil(bottom-2.5));
    if (x1 <= x0 || y1 <= y0) return result;
    int step = (int)fmax(1, ceil(sqrt((double)(x1-x0)*(y1-y0)/65536.0)));
    while (ceil((double)(x1-x0)/step)*ceil((double)(y1-y0)/step) > 65536) step++;
    unsigned *histogram = calloc(65536, sizeof(unsigned));
    if (!histogram) { result.status = "unavailable"; return result; }
    const int derivative[] = {-1,-2,0,2,1}, smooth[] = {1,4,6,4,1};
    double mean = 0, m2 = 0, energy = 0;
    for (int by = y0; by < y1; by += step) {
        for (int bx = x0; bx < x1; bx += step) {
            /* Fixed stratified jitter prevents a regular grid from repeatedly
             * missing thin periodic edges. Stable across frames of equal size. */
            const uint32_t hash = (uint32_t)bx*73856093u ^ (uint32_t)by*19349663u;
            const int cell_w = (int)fmin(step, x1-bx), cell_h = (int)fmin(step, y1-by);
            const int x = bx + (int)(hash % (unsigned)cell_w);
            const int y = by + (int)((hash >> 16) % (unsigned)cell_h);
            if (roi.circle) {
                const double dx = fabs(x+.5-roi.a*width)+2;
                const double dy = fabs(y+.5-roi.b*height)+2;
                if (dx*dx+dy*dy > radius*radius) continue;
            }
            double gx = 0, gy = 0;
            unsigned center = 0, a = 0, b = 0, c = 0;
            for (int j = 0; j < 5; j++) for (int i = 0; i < 5; i++) {
                const unsigned value = sample(context, x+i-2, y+j-2);
                gx += value * (double)(derivative[i]*smooth[j]);
                gy += value * (double)(smooth[i]*derivative[j]);
                if (i == 2 && j == 2) center = value;
                if (i == 3 && j == 2) a = value;
                if (i == 2 && j == 3) b = value;
                if (i == 3 && j == 3) c = value;
            }
            const unsigned diagonal = (unsigned)abs((int)center-(int)a-(int)b+(int)c);
            histogram[(diagonal+1)/2]++;
            result.count++;
            const double delta = center-mean;
            mean += delta/result.count;
            m2 += delta*(center-mean);
            energy += (gx*gx+gy*gy)/16384.0;
        }
    }
    unsigned cumulative = 0, median = 0;
    for (; median < 65535; median++) {
        cumulative += histogram[median];
        if (cumulative >= (result.count+1)/2) break;
    }
    free(histogram);
    if (result.count < 64) return result;
    const double noise = median/.67448975, noise2 = noise*noise;
    const double signal = m2/result.count-noise2;
    result.status = "low-detail";
    if (signal <= fmax(maximum*maximum*.000025, 4*noise2)) return result;
    const double corrected = fmax(0, energy/result.count - (1400.0/16384)*noise2);
    if (corrected <= 0) return result;
    result.score = 1000*corrected/signal;
    result.status = "ok";
    return result;
}
