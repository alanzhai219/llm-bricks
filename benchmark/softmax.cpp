#include "common.hpp"
#include "micro/softmax.hpp"

namespace bench {

void softmax(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    Tensor x(options.shape), out(options.shape);
    fill(x);
    SoftmaxParams params(x, out);
    Softmax op(context);
    op.set_input(params);
    measure(options, out, [] {}, [&] { return op.infer(); });
}

}  // namespace bench