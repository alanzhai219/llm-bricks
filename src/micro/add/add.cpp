#include "micro/add.hpp"
#include "add_ref.hpp"

namespace llm_bricks {
Status Add::infer() {
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    switch (m_ctx.backend) {
    case Backend::automatic:
    case Backend::reference: return detail::add_ref(*m_params);
    default: return Status::unsupported;
    }
}
}  // namespace llm_bricks