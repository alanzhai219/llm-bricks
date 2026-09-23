#include "micro/gelu.hpp"
#include "gelu_avx2.hpp"
#include "gelu_ref.hpp"

namespace llm_bricks {
Status Gelu::infer() {
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    switch (m_ctx.backend) {
    case Backend::automatic:
    case Backend::reference: return detail::gelu_ref(*m_params);
    case Backend::avx2: return detail::gelu_avx2(*m_params);
    default: return Status::unsupported;
    }
}
}  // namespace llm_bricks