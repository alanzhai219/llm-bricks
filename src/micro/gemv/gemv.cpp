#include "micro/gemv.hpp"
#include "gemv_ref.hpp"
#include "gemv_avx2.hpp"

namespace llm_bricks {
Status Gemv::infer() {
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    switch (m_ctx.backend) {
    case Backend::automatic:
    case Backend::reference:
        return detail::gemv_ref(*m_params);
    case Backend::avx2:
        return detail::gemv_avx2(*m_params);
    default:
        return Status::unsupported;
    }
}
}  // namespace llm_bricks