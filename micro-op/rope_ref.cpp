#include "tensor.hpp"
#include "rope.hpp"

void rope_ref(Tensor4D& x, float base) {
    const auto x_shape = x.get_shape();
    const size_t B = x_shape[0];
    const size_t H = x_shape[1];
    const size_t S = x_shape[2];
    const size_t D = x_shape[3];

    if (D % 2 != 0) {
        throw std::runtime_error("D must be even");
    }

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                // --------------------------------------------
                // position = s
                // --------------------------------------------
                const size_t position = s;
                for (size_t d = 0; d < D; d += 2) {
                    // ----------------------------------------
                    // frequency
                    //
                    // inv_freq = 1 / base^(d / D)
                    // ----------------------------------------
                    float exponent = static_cast<float>(d) / static_cast<float>(D);
                    float inv_freq = 1.0f / std::pow(base, exponent);

                    // ----------------------------------------
                    // rotation angle
                    // ----------------------------------------
                    float theta = position * inv_freq;

                    float c = std::cos(theta);
                    float s = std::sin(theta);

                    // ----------------------------------------
                    // load pair
                    // ----------------------------------------
                    float x0 = x.at(b, h, s, d);
                    float x1 = x.at(b, h, s, d + 1);

                    // ----------------------------------------
                    // rotate
                    // x0' = x0*c - x1*s
                    // x1' = x0*s + x1*c
                    // ----------------------------------------
                    float y0 = x0 * c - x1 * s;
                    float y1 = x0 * s + x1 * c;

                    // ----------------------------------------
                    // store
                    // ----------------------------------------
                    x.at(b, h, s, d)     = y0;
                    x.at(b, h, s, d + 1) = y1;
                }
            }
        }
    }
}

