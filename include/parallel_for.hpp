#pragma once

#include <omp.h>

template <typename F>
void parallel_nt(int nthr, const F& func) {
    if (nthr == 0) nthr = omp_get_max_threads();
    if (nthr == 1) { func(0, 1); return; }

    int origin_dyn_val = omp_get_dynamic();
    if (origin_dyn_val != 0) omp_set_dynamic(0);

#pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        int nthr_act = omp_get_num_threads();
        func(ithr, nthr_act);
    }

    if (origin_dyn_val != 0) omp_set_dynamic(origin_dyn_val);
}

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }
    n_end += n_start;
}

/*
*
void vector_add(const float* A, const float* B, float* C, size_t N, int nthr = 0) {
    parallel_nt(nthr, [&](int ithr, int nthr_act) {
        size_t start, end;
        splitter(N, nthr_act, ithr, start, end);

        // 主循环（可选手动向量化提示, 告诉编译期生成可向量化的代码）
#pragma omp simd
        for (size_t i = start; i < end; ++i) {
            C[i] = A[i] + B[i];
        }
    });
}
 * */
