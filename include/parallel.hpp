#pragma once

#include <algorithm>
#include <cstddef>
#include <omp.h>

namespace llm_bricks {

// 小 workload 不启动 OpenMP。
// 这个值最终应该通过 benchmark 调整。
constexpr std::size_t DEFAULT_GRAIN_SIZE = 32;

// ============================================================
// splitter
//
// 将 [0, n) 均匀划分给 team 个线程。
//
// n = 10, team = 3:
//
// tid 0 -> [0, 4)
// tid 1 -> [4, 7)
// tid 2 -> [7, 10)
//
// Note: 每个线程 workload 差 <= 1 !!!
// ============================================================

template <typename T>
inline void splitter(T n, int team, int tid, T& start, T& end) {
    if (n == 0) {
        start = 0;
        end = 0;
        return;
    }

    if (team <= 1) {
        start = 0;
        end = n;
        return;
    }

    const T q = n / static_cast<T>(team);
    const T r = n % static_cast<T>(team);

    const T t = static_cast<T>(tid);

    if (t < r) {
        // 前 r 个线程多拿一个元素
        start = t * (q + 1);
        end   = start + (q + 1);
    } else {
        start = r * (q + 1) + (t - r) * q;
        end = start + q;
    }
}


// ============================================================
// parallel_nt
// Low-level primitive.
// func(ithr, nthr)
//
// ithr    : 当前线程 ID
// nthr    : 实际 OpenMP thread 数
//
// 注意：这里不使用 omp for。
// work partition 由上层代码控制。
// ============================================================

template <typename F>
inline void parallel_nt(int nthr, const F& func) {
    if (nthr <= 0) {
        nthr = omp_get_max_threads();
    }

    nthr = std::max(1, nthr);

    // 避免进入 OpenMP runtime。
    if (nthr == 1) {
        func(0, 1);
        return;
    }

#pragma omp parallel num_threads(nthr)
    {
        const int ithr = omp_get_thread_num();
        const int nthr_act = omp_get_num_threads();

        func(ithr, nthr_act);
    }
}

} // namespace llm_bricks  
