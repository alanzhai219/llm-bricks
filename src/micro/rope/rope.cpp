#include "micro/rope.hpp"
#include "rope_ref.hpp"

namespace llm_bricks {

Status rope(Context& context, const RopeParams& params) {
    if (context.device.type != DeviceType::cpu) return Status::unsupported;

    switch (context.backend) {
    case Backend::automatic:
    case Backend::reference:
        return detail::rope_ref(params);
    case Backend::avx2:
    case Backend::avx512:
    case Backend::cuda:
    case Backend::sycl:
        return Status::unsupported;
    }

    return Status::unsupported;
}

} // namespace llm_bricks
