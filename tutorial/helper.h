#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_mat(float* mat, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            printf("%f ", mat[j * n_rows + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_rand(float* mat, size_t n_elems) {
    for (size_t i = 0; i < n_elems; i++) {
        *mat++ = rand() / (float)RAND_MAX;
    }
}

void init_const(float* mat, float value, size_t n_elems) {
    for (size_t i = 0; i < n_elems; i++) {
        *mat++ = value;
    }
}

void validate_mat(float* mat, float* mat_ref, size_t n_elems, float eps) {
    for (size_t i = 0; i < n_elems; i++) {
        float value = mat[i];
        float ref_value = mat_ref[i];
        if (isnan(value)) {
            printf("Error, NAN found!\n");
            return;
        } else if (isinf(value)) {
            printf("Error, INF found!\n");
            return;
        } else {
            if (fabsf((value - ref_value) / ref_value) > eps) {
                printf("Error, %f != %f\n", value, ref_value);
                return;
            }
        }
    }
    printf("PASSED!\n");
}

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}