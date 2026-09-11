#include <cmath>
#include <vector>

#include "micro/rope.hpp"
#include "test.hpp"

void setup_rope_test_data(llm_bricks::Tensor& x) {
    float* data = x.data_as<float>();
    data[0] = 1.0F;
    data[1] = 2.0F;
    data[2] = 3.0F;
    data[3] = 4.0F;
    data[4] = 1.0F;
    data[5] = 2.0F;
    data[6] = 3.0F;
    data[7] = 4.0F;
}

void setup_expected_result(llm_bricks::Tensor& result) {
    float* data = result.data_as<float>();
    data[0] = 1.0F;
    data[1] = 2.0F;
    data[2] = 3.0F;
    data[3] = 4.0F;
    data[4] = std::cos(1.0F) - 2.0F * std::sin(1.0F);
    data[5] = std::sin(1.0F) + 2.0F * std::cos(1.0F);
    const float second_pair_theta = std::pow(10000.0F, -0.5F);
    data[6] = 3.0F * std::cos(second_pair_theta) - 4.0F * std::sin(second_pair_theta);
    data[7] = 3.0F * std::sin(second_pair_theta) + 4.0F * std::cos(second_pair_theta);
}

int test_rope_ref() {
    llm_bricks::Tensor x({1, 1, 2, 4});
    setup_rope_test_data(x);

    llm_bricks::RopeParams params{x, 10000.0F, 4, true};
    llm_bricks::Context context;
    // runtime check
    CHECK(llm_bricks::rope(context, params) == llm_bricks::Status::ok);

    llm_bricks::Tensor expected({1, 1, 2, 4});
    setup_expected_result(expected);

    float* data = x.data_as<float>();
    float* expected_data = expected.data_as<float>();
    // result check
    for (size_t i = 0; i < x.numel(); ++i) {
        CHECK_NEAR(data[i], expected_data[i], 1.0e-6F);
    }
    printf("\033[32m## ROPE PASS ##\033[0m\n");
    return 0;
}

int main() {
    auto result = test_rope_ref();
    return result;
}