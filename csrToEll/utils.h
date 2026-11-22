#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdio>

inline bool checkResult(const float *ref, const float *out, int n, float tol = 1e-4) {
    for (int i = 0; i < n; i++) {
        if (std::abs(ref[i] - out[i]) > tol * (1.0f + std::abs(ref[i]))) {
            printf("FAIL at index %d: ref=%f, out=%f\n", i, ref[i], out[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

#endif // UTILS_H