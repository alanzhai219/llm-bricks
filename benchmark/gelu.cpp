#include <algorithm>

#include "common.hpp"
#include "micro/gelu.hpp"

namespace bench {

void gelu(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    Tensor x(options.shape);
    fill(x);
    Tensor original(x);
    GeluParams params(x);
    Gelu op(context);
    op.set_input(params);
    measure(options, x, [&] { std::copy_n(original.data(), x.numel(), x.data()); }, [&] { return op.infer(); });
}

}  // namespace bench