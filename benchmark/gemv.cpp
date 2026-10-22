#include <algorithm>

#include "common.hpp"
#include "micro/gemv.hpp"

namespace bench {

void gemv(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    size_t m = options.shape[0], n = options.shape[1];
    Tensor a({m, n});
    fill(a);
    std::vector<float> b_data(n, 1.0F);
    Tensor y({m});
    GemvParams params(a, b_data.data(), y);
    Gemv op(context);
    op.set_input(params);
    measure(options, y, [&] { std::fill_n(y.data(), y.numel(), 0.0F); }, [&] { return op.infer(); });
}

}  // namespace bench