#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int get_niter(int matsize, int niter_start, int niter_end, int matsize_start, int matsize_end) {

    if (matsize_end == matsize_start || niter_start == niter_end) return niter_start;
    float a = ((float)niter_end - niter_start) * (matsize_start * matsize_end)
              / (matsize_start - matsize_end);
    float b = niter_start - a / matsize_start;
    return round(a / matsize + b);
}

void init_rand(float* mat, size_t n_elem) {
    for (size_t i = 0; i < n_elem; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

void init_const(float* mat, float value, size_t n_elem) {
    for (size_t i = 0; i < n_elem; i++) {
        mat[i] = value;
    }
}

struct val_stat_t {
    int n_error;
    int n_nans;
    int n_inf;
};

struct val_stat_t validate_mat(float* mat, float* mat_ref, size_t n_elem, float eps) {
    struct val_stat_t result = {0, 0, 0};
    for (size_t i = 0; i < n_elem; i++) {
        float value = mat[i];
        float value_ref = mat_ref[i];
        if (isnan(value)) {
            result.n_nans += 1;
            result.n_error += 1;
            continue;
        }
        if (isinf(value)) {
            result.n_inf += 1;
            result.n_error += 1;
            continue;
        }
        if (fabsf((value - value_ref) / value_ref) > eps) {
            result.n_error += 1;
            continue;
        }
    }
    return result;
}

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

void printfn(const char* str, int n) {
    for (int i = 0; i < n; i++) {
        printf("%s", str);
    }
}