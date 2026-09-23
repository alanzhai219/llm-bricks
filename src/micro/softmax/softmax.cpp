#include "micro/softmax.hpp"
#include "softmax_ref.hpp"

namespace llm_bricks {
Status Softmax::infer() {
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    switch (m_ctx.backend) {
    case Backend::automatic:
    case Backend::reference: return detail::softmax_ref(*m_params);
    default: return Status::unsupported;
    }
}
}  // namespace llm_bricks