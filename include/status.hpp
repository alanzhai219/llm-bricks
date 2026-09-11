#pragma once

namespace llm_bricks {

enum class Status {
    ok = 0,
    invalid_argument,
    unsupported,
};

[[nodiscard]] constexpr bool succeeded(Status status) {
    return status == Status::ok;
}

} // namespace llm_bricks
