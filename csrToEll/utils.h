#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdio>

// made inline to prevent "multiple definition" linker errors
inline bool checkResult(const float *ref, const float *out, int n, float tol = 1e-4) {
    for (int i = 0; i < n; i++) {
        // using absolute difference to handle floating-point inaccuracies
        if (std::abs(ref[i] - out[i]) > tol) {
            printf("FAIL at index %d: ref=%f, out=%f\n", i, ref[i], out[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

#endif // UTILS_H