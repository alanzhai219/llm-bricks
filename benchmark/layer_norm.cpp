#include "common.hpp"
#include "micro/layer_norm.hpp"

namespace bench {

void layer_norm(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    Tensor x(options.shape), gamma({options.shape.back()}, 1.0F);
    Tensor beta({options.shape.back()}, 0.0F), out(options.shape);
    fill(x);
    LayerNormParams params(x, gamma, beta, 1e-5F, out);
    LayerNorm op(context);
    op.set_input(params);
    measure(options, out, [] {}, [&] { return op.infer(); });
}

}  // namespace bench