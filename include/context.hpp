#pragma once

namespace llm_bricks {

enum class DeviceType {
    cpu,
    cuda,
    sycl,
};

enum class Backend {
    automatic,
    reference,
    avx2,
    avx512,
    cuda,
    sycl,
};

struct Context {
    DeviceType device = DeviceType::cpu;
    Backend backend = Backend::automatic;
};

} // namespace llm_bricks
