#include "macro/token_embd.hpp"
#include "token_embd_ref.hpp"
#include "token_embd_opt.hpp"

namespace llm_bricks {
Status TokenEmbedding::infer() {
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }
    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }
    switch (m_ctx.backend) {
    case Backend::automatic:
    case Backend::reference: return detail::token_embd_ref(*m_params);
    case Backend::x86_opt: return detail::token_embd_opt(*m_params);
    default: return Status::unsupported;
    }
}
} // namespace llm_bricks