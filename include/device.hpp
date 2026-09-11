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

struct Device {
    DeviceType type = DeviceType::cpu;
    int index = 0;
};

} // namespace llm_bricks
