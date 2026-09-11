#include "micro/rope.hpp"

#include <cmath>

namespace llm_bricks::detail {

Status rope_ref(TensorView x, const RopeParams& params) {
    if (x.data == nullptr || x.dtype != DType::f32 || x.device.type != DeviceType::cpu ||
            x.ndim != 4 || !x.is_contiguous() || params.base <= 0.0F) {
        return Status::invalid_argument;
    }

    const std::size_t batch = static_cast<std::size_t>(x.shape[0]);
    const std::size_t heads = static_cast<std::size_t>(x.shape[1]);
    const std::size_t sequence = static_cast<std::size_t>(x.shape[2]);
    const std::size_t head_size = static_cast<std::size_t>(x.shape[3]);
    const std::size_t rotary_dim = params.rotary_dim == 0 ? head_size : params.rotary_dim;

    if (rotary_dim == 0 || rotary_dim > head_size || rotary_dim % 2 != 0) {
        return Status::invalid_argument;
    }

    auto* data = x.data_as<float>();
    const std::size_t half = rotary_dim / 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            for (std::size_t position = 0; position < sequence; ++position) {
                float* token = data + (((b * heads + h) * sequence + position) * head_size);
                for (std::size_t pair = 0; pair < half; ++pair) {
                    const float exponent = static_cast<float>(2 * pair) /
                            static_cast<float>(rotary_dim);
                    const float theta = static_cast<float>(position) * std::pow(params.base, -exponent);
                    const float cosine = std::cos(theta);
                    const float sine = std::sin(theta);
                    const std::size_t first = params.interleaved ? 2 * pair : pair;
                    const std::size_t second = params.interleaved ? first + 1 : pair + half;
                    const float x0 = token[first];
                    const float x1 = token[second];
                    token[first] = x0 * cosine - x1 * sine;
                    token[second] = x0 * sine + x1 * cosine;
                }
            }
        }
    }

    return Status::ok;
}

} // namespace llm_bricks::detail
