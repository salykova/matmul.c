#pragma once
#include <math.h>
#include <stdlib.h>

int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int scedule_niter(int matsize, int niter_start, int niter_end, int matsize_start, int matsize_end) {
    float a = ((float)niter_end - niter_start) * (matsize_start * matsize_end)
              / (matsize_start - matsize_end);
    float b = niter_start - a / matsize_start;
    return round(a / matsize + b);
}

void init_rand(float* mat, int n_elem) {
    for (int i = 0; i < n_elem; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

void init_const(float* mat, float value, int n_elem) {
    for (int i = 0; i < n_elem; i++) {
        mat[i] = value;
    }
}

struct val_data {
    int n_error;
    int n_nans;
    int n_inf;
};

struct val_data validate_mat(float* mat, float* mat_ref, int n_elem, float eps) {
    struct val_data result = {0, 0, 0};
    for (int i = 0; i < n_elem; i++) {
        float value = mat[i];
        float value_ref = mat_ref[i];
        if (isnan(value)) {
            result.n_nans += 1;
            continue;
        }
        if (isinf(value_ref)) {
            result.n_inf += 1;
            continue;
        }
        if (fabsf(value - value_ref) / value_ref > eps) {
            result.n_error += 1;
            continue;
        }
    }
    return result;
}
