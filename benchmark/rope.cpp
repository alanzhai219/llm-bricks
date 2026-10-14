#include <algorithm>
#include <stdexcept>

#include "common.hpp"
#include "micro/rope.hpp"

namespace bench {

void rope(const Options& options) {
    using namespace llm_bricks;
    if (options.shape.size() != 4 || options.shape[3] % 2 != 0) {
        throw std::invalid_argument("rope expects B,H,S,D with even D");
    }
    Context context(DeviceType::cpu, options.backend);
    Tensor x(options.shape);
    fill(x);
    Tensor original(x);
    RopeParams params(x);
    Rope op(context);
    op.set_input(params);
    measure(options, x, [&] { std::copy_n(original.data(), x.numel(), x.data()); }, [&] { return op.infer(); });
}

}  // namespace bench