/* Times original Darknet's network_predict with the same protocol as `cuda_detector benchmark
 * --scope network`: letterboxed input prepared once, `warmup` untimed calls, then `iterations`
 * timed calls of input upload + forward + head download, reported as mean/p50/p95/min/max ms.
 * Uses only Darknet's public API (include/darknet.h); build against an unmodified libdarknet.a:
 *   gcc -O2 -DGPU -I<darknet>/include -I/usr/local/cuda/include darknet_bench.c <darknet>/libdarknet.a \
 *       -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++ -lm -pthread -o darknet_bench
 */
#include "darknet.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef GPU
#include <cuda_runtime.h>
#endif

/* Monotonic wall clock in milliseconds. */
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* qsort comparator for ascending doubles. */
static int ascending(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Usage: darknet_bench cfg weights image warmup iterations; prints one result line. */
int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <cfg> <weights> <image> <warmup> <iterations>\n", argv[0]);
        return 2;
    }
    int warmup = atoi(argv[4]), iterations = atoi(argv[5]);
    if (warmup < 0 || iterations < 1) return 2;
    network* net = load_network(argv[1], argv[2], 0);
    set_batch_network(net, 1);
    image source = load_image_color(argv[3], 0, 0);
    image sized = letterbox_image(source, net->w, net->h);
    for (int i = 0; i < warmup; ++i) network_predict(net, sized.data);
    double* times = calloc((size_t)iterations, sizeof(double));
    double total = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double begin = now_ms();
        network_predict(net, sized.data);
#ifdef GPU
        cudaDeviceSynchronize();
#endif
        times[i] = now_ms() - begin;
        total += times[i];
    }
    qsort(times, (size_t)iterations, sizeof(double), ascending);
    printf("darknet iterations=%d mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f min_ms=%.3f max_ms=%.3f\n", iterations,
           total / iterations, times[(iterations - 1) / 2], times[(int)((iterations - 1) * 0.95)], times[0],
           times[iterations - 1]);
    free(times);
    free_image(sized);
    free_image(source);
    return 0;
}
