#ifndef INSPECTION_FOCUS_H
#define INSPECTION_FOCUS_H
#include <stdint.h>

typedef struct {
    int circle;
    double a, b, c, d; /* normalized rectangle bounds or circle cx, cy, r, 0 */
} FocusRoi;
typedef struct {
    double score;
    unsigned count;
    const char *status; /* ok, too-small, low-detail */
} FocusResult;
typedef uint16_t (*FocusSample)(const void *context, int x, int y);
FocusResult measure_focus(int width, int height, double maximum, FocusRoi roi,
                          FocusSample sample, const void *context);
#endif
