#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "context.hpp"
#include "linux_perf.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace bench {

struct Options {
    std::string op;
    std::vector<size_t> shape;
    llm_bricks::Backend backend = llm_bricks::Backend::reference;
    size_t warmup = 5;
    size_t iters = 20;
    bool perf = false;
};

inline void fill(llm_bricks::Tensor& tensor) {
    for (size_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125F;
    }
}

// setup is outside the timed interval (in-place ops and matmul need a reset).
template <typename Setup, typename Infer>
void measure(const Options& options, const llm_bricks::Tensor& result, Setup setup, Infer infer) {
    auto run = [&] {
        if (infer() != llm_bricks::Status::ok) {
            throw std::runtime_error("infer() failed: check shape and backend support");
        }
    };
    for (size_t i = 0; i < options.warmup; ++i) {
        setup();
        run();
    }

    std::vector<double> times;
    times.reserve(options.iters);
    for (size_t i = 0; i < options.iters; ++i) {
        setup();
        const auto start = std::chrono::steady_clock::now();
        if (options.perf) {
            setup();
            auto scope = llm_bricks::LinuxPerf::Profile(options.op, i);
            run();
        } else {
            run();
        }
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    double sum = 0;
    for (double time : times) sum += time;
    std::cout << std::fixed << std::setprecision(3)
              << "min=" << times.front() << " us  median=" << times[times.size() / 2]
              << " us  mean=" << sum / times.size() << " us"
              << "  output[0]=" << result.data()[0] << '\n';

    if (options.perf) {
        setup();
        auto scope = llm_bricks::LinuxPerf::Profile(options.op);
        run();
    }
}

void add(const Options& options);
void gelu(const Options& options);
void layer_norm(const Options& options);
void matmul(const Options& options);
void rope(const Options& options);
void scale(const Options& options);
void softmax(const Options& options);
void gemv(const Options& options);
}  // namespace bench