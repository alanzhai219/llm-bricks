#include <algorithm>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "micro/matmul.hpp"

namespace bench {

void matmul(const Options& options) {
    using namespace llm_bricks;
    const auto& shape = options.shape;
    if (shape.size() != 3 && shape.size() != 4) {
        throw std::invalid_argument("matmul expects M,K,N or B,M,K,N");
    }
    const size_t end = shape.size();
    const size_t m = shape[end - 3], k = shape[end - 2], n = shape[end - 1];
    const bool batched = end == 4;
    Context context(DeviceType::cpu, options.backend);
    Tensor a(batched ? std::vector<size_t>{shape[0], m, k} : std::vector<size_t>{m, k});
    Tensor b(batched ? std::vector<size_t>{shape[0], k, n} : std::vector<size_t>{k, n});
    Tensor c(batched ? std::vector<size_t>{shape[0], m, n} : std::vector<size_t>{m, n});
    fill(a);
    fill(b);
    MatMulParams params(a, b, c);
    MatMul op(context);
    op.set_input(params);
    measure(options, c, [&] { std::fill_n(c.data(), c.numel(), 0.0F); }, [&] { return op.infer(); });
}

}  // namespace bench