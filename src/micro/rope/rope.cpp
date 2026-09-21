#include "micro/rope.hpp"
#include "rope_ref.hpp"

namespace llm_bricks {

Status Rope::infer() {
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }

    switch (m_ctx.backend) {
        case Backend::automatic:
        case Backend::reference:
            return detail::rope_ref(*m_params);
        case Backend::avx2:
            return Status::unsupported;
        case Backend::avx512:
            return Status::unsupported;
        case Backend::cuda:
            return Status::unsupported;
        case Backend::sycl:
            return Status::unsupported;
    }

    return Status::unsupported;
}

} // namespace llm_bricks
