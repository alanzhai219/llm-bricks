#include "micro/matmul.hpp"
#include "matmul_ref.hpp"

namespace llm_bricks {

Status MatMul::infer() {
    // Implement the matrix multiplication logic here
    if (m_params == nullptr) {
        return Status::invalid_argument;
    }

    if (m_ctx.device != DeviceType::cpu) {
        return Status::unsupported;
    }

    switch (m_ctx.backend) {
        case Backend::automatic:
        case Backend::reference:
            return detail::matmul_ref(*m_params);
        case Backend::avx2:
            // Implement AVX2 optimized matrix multiplication here
            return Status::unsupported;
        case Backend::avx512:
            // Implement AVX512 optimized matrix multiplication here
            return Status::unsupported;
        case Backend::cuda:
            // Implement CUDA optimized matrix multiplication here
            return Status::unsupported;
        case Backend::sycl:
            // Implement SYCL optimized matrix multiplication here
            return Status::unsupported;
        default:
            return Status::unsupported;

    }
}

}  // namespace llm_bricks