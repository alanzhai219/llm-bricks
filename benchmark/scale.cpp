#include "common.hpp"
#include "micro/scale.hpp"

namespace bench {

void scale(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    Tensor x(options.shape), out(options.shape);
    fill(x);
    ScaleParams params(x, 0.5F, out);
    Scale op(context);
    op.set_input(params);
    measure(options, out, [] {}, [&] { return op.infer(); });
}

}  // namespace bench