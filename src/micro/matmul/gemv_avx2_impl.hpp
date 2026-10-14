#pragma once
#include <immintrin.h>

__attribute__((target("avx2,fma")))
inline void gemv_avx2_v1(const float* a, const float* b, float* c,
                         size_t k, size_t n) {
    const size_t vector_end = n & ~size_t{7};
    for (size_t p = 0; p < k; ++p) {
        const __m256 av = _mm256_set1_ps(a[p]);
        const float* b_row = b + p * n;
        size_t j = 0;
        for (; j < vector_end; j += 8) {
            const __m256 cv = _mm256_loadu_ps(c + j);
            const __m256 bv = _mm256_loadu_ps(b_row + j);
            _mm256_storeu_ps(c + j, _mm256_fmadd_ps(av, bv, cv));
        }
        for (; j < n; ++j) {
            c[j] += a[p] * b_row[j];
        }
    }
}