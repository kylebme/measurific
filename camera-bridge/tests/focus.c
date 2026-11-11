#include "../src/focus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(expression) do { if (!(expression)) { fprintf(stderr, "Failed: %s (line %d)\n", #expression, __LINE__); exit(1); } } while (0)
typedef struct {
    int width, height;
    double angle, sigma, gain, offset, maximum;
    int mode;
    FocusRoi roi;
} Scene;

static uint16_t sample(const void *context, int x, int y) {
    const Scene *scene = context;
    CHECK(x >= 0 && y >= 0 && x < scene->width && y < scene->height);
    /* Fail if even one neighborhood pixel is outside the actual ROI. */
    const FocusRoi roi = scene->roi;
    if (roi.circle) {
        const double dx = x+.5-roi.a*scene->width, dy = y+.5-roi.b*scene->height;
        const double r = roi.c*fmin(scene->width, scene->height);
        CHECK(dx*dx+dy*dy <= r*r+1e-9);
    } else {
        CHECK(x+.5 >= roi.a*scene->width && x+.5 < roi.c*scene->width);
        CHECK(y+.5 >= roi.b*scene->height && y+.5 < roi.d*scene->height);
    }
    double value = .5;
    if (scene->mode == 1) {
        /* Deterministic pseudo-random flat-field noise. */
        uint32_t h = (uint32_t)x*374761393u + (uint32_t)y*668265263u;
        h = (h^(h >> 13))*1274126177u; h ^= h >> 16;
        value += .08*((double)(h & 65535)/65535-.5);
    } else if (scene->mode == 0) {
        const double distance = (x+.5-scene->width/2.0)*cos(scene->angle) + (y+.5-scene->height/2.0)*sin(scene->angle);
        value = .1+.4*(1+erf(distance/(sqrt(2)*scene->sigma)));
    }
    return (uint16_t)lround(scene->maximum*(scene->gain*value+scene->offset));
}

int main(void) {
    const FocusRoi full = {0, 0, 0, 1, 1};
    Scene scene = {128, 128, 0, .5, 1, 0, 65535, 0, full};
    double vertical = 0;
    for (int orientation = 0; orientation < 3; orientation++) {
        scene.angle = orientation*.7853981633974483;
        double previous = 1e9;
        const double sigmas[] = {.5, 1, 2, 4, 8};
        for (int i = 0; i < 5; i++) {
            scene.sigma = sigmas[i];
            FocusResult result = measure_focus(scene.width, scene.height, scene.maximum, full, sample, &scene);
            CHECK(!strcmp(result.status, "ok"));
            CHECK(result.score > 0 && result.score < previous);
            previous = result.score;
            if (orientation == 0 && i == 0) vertical = result.score;
            if (orientation == 2 && i == 0) CHECK(fabs(result.score-vertical) < .001);
        }
    }
    scene.angle = 0; scene.sigma = .001;
    CHECK(fabs(measure_focus(128,128,65535,full,sample,&scene).score - 1000*.2/124/.16) < .002);
    scene.sigma = 2;
    const double original = measure_focus(128,128,65535,full,sample,&scene).score;
    scene.gain = .5; scene.offset = .1;
    CHECK(fabs(measure_focus(128,128,65535,full,sample,&scene).score/original-1) < .001);
    scene.gain = 1; scene.offset = 0; scene.maximum = 255;
    CHECK(fabs(measure_focus(128,128,255,full,sample,&scene).score/original-1) < .01);
    scene.maximum = 65535;
    scene.mode = 2;
    CHECK(!strcmp(measure_focus(128,128,65535,full,sample,&scene).status, "low-detail"));
    scene.mode = 1;
    CHECK(!strcmp(measure_focus(128,128,65535,full,sample,&scene).status, "low-detail"));
    scene.mode = 0;
    const FocusRoi regions[] = {{0,.213,.319,.793,.717}, {1,.6,.45,.2,0}};
    for (int i=0; i<2; i++) {
        scene.roi = regions[i];
        FocusResult result = measure_focus(128,128,65535,scene.roi,sample,&scene);
        CHECK(!strcmp(result.status, "ok"));
    }
    const FocusRoi tiny = {0,.5,.5,.52,.52};
    CHECK(!strcmp(measure_focus(128,128,65535,tiny,sample,&scene).status, "too-small"));
    scene.roi = full; scene.width = 4096; scene.height = 4096;
    const clock_t start = clock();
    FocusResult large = measure_focus(4096,4096,65535,full,sample,&scene);
    CHECK(large.count <= 65536 && !strcmp(large.status, "ok"));
    printf("Native focus tests passed; 4096x4096 / %u centers: %.1f ms (includes synthetic erf rendering)\n", large.count, 1000.0*(clock()-start)/CLOCKS_PER_SEC);
    scene.width = 10; scene.height = 100000;
    CHECK(measure_focus(scene.width,scene.height,65535,full,sample,&scene).count <= 65536);
    return 0;
}
