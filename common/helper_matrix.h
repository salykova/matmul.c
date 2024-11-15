#pragma once
#include <math.h>
#include <stdlib.h>

void init_rand(float* mat, const int m, const int n) {
    for (int i = 0; i < m * n; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

void init_const(float* mat, const float value, const int m, const int n) {
    for (int i = 0; i < m * n; i++) {
        mat[i] = value;
    }
}

struct cmp_result {
    int n_false;
    int n_nans;
    int n_inf;
};

struct cmp_result compare_mats(float* mat, float* mat_ref, const int m, const int n) {
    struct cmp_result result = {0, 0, 0};
    for (int i = 0; i < m * n; i++) {
        float value = mat[i];
        float value_ref = mat_ref[i];
        if (isnan(value)) {
            result.n_nans += 1;
            result.n_false += 1;
        }
        if (isinf(value_ref)) {
            result.n_inf += 1;
            result.n_false += 1;
        }
        if (fabsf(value - value_ref) > 1e-3) {
            result.n_false += 1;
            return result;
        }
    }

    return result;
}
