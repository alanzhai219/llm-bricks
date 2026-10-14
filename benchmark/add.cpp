#include "common.hpp"
#include "micro/add.hpp"

namespace bench {

void add(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    Tensor x(options.shape), y(options.shape), out(options.shape);
    fill(x);
    fill(y);
    AddParams params(x, y, out);
    Add op(context);
    op.set_input(params);
    measure(options, out, [] {}, [&] { return op.infer(); });
}

}  // namespace bench