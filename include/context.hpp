#pragma once

#include "device.hpp"

namespace llm_bricks {

struct Context {
    Device device {};
    Backend backend = Backend::automatic;
};

} // namespace llm_bricks
