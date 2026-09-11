#include "micro/rope.hpp"

#include <cmath>

namespace llm_bricks::detail {

Status rope_ref(const RopeParams& params) {
    auto x = params.x;
    if (x.ptr() == nullptr || x.dtype() != DType::f32 || x.ndims() != 4 || params.base <= 0.0F) {
        return Status::invalid_argument;
    }

    if (params.interleaved != true || params.rotary_dim % 2 != 0) {
        return Status::invalid_argument;
    }

    const auto B = x.dim(0);
    const auto H = x.dim(1);
    const auto S = x.dim(2);
    const auto D = x.dim(3);

    const auto rotary_dim = params.rotary_dim == 0 ? D : params.rotary_dim;

    if (rotary_dim == 0 || rotary_dim > D || rotary_dim % 2 != 0) {
        return Status::invalid_argument;
    }

    auto* data = static_cast<float*>(x.ptr());
    const size_t half = rotary_dim / 2;
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t position = 0; position < S; ++position) {
                float* token = data + (((b * H + h) * S + position) * D);
                for (size_t pair = 0; pair < half; ++pair) {
                    // compute the rotation angle for this pair of dimensions
                    const float exponent = static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
                    const float theta    = static_cast<float>(position) * std::pow(params.base, -exponent);
                    // compute the cosine and sine of the rotation angle
                    const float cos_ = std::cos(theta);
                    const float sin_ = std::sin(theta);
                    // get the indices of the two elements to rotate
                    const size_t first  = params.interleaved ? 2 * pair  : pair;
                    const size_t second = params.interleaved ? first + 1 : pair + half;
                    const float x0 = token[first];
                    const float x1 = token[second];
                    // apply the rotation
                    token[first]  = x0 * cos_ - x1 * sin_;
                    token[second] = x0 * sin_ + x1 * cos_;
                }
            }
        }
    }

    return Status::ok;
}

} // namespace llm_bricks::detail
