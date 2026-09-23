#include "micro/scale.hpp"
#include "scale_avx2.hpp"
#include "scale_ref.hpp"

namespace llm_bricks {
Status Scale::infer() {
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    switch (m_ctx.backend) {
    case Backend::automatic:
    case Backend::reference: return detail::scale_ref(*m_params);
    case Backend::avx2: return detail::scale_avx2(*m_params);
    default: return Status::unsupported;
    }
}
}  // namespace llm_bricks