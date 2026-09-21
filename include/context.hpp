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
    Context() = default;
    virtual ~Context() {}
    Context(const Context& ctx) {
        device = ctx.device;
        backend = ctx.backend;
    }
    Context(Context&& ctx) {
        device = ctx.device;
        backend = ctx.backend;
    }
    Context& operator=(const Context& ctx) {
        device = ctx.device;
        backend = ctx.backend;
        return *this;
    }
    Context& operator=(Context&& ctx) {
        device = ctx.device;
        backend = ctx.backend;
        return *this;
    }
    DeviceType device = DeviceType::cpu;
    Backend backend = Backend::automatic;
};

} // namespace llm_bricks
