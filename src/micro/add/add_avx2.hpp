#pragma once

#include "micro/add.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))

#include <immintrin.h>

namespace llm_bricks::detail {

__attribute__((target("avx2"))) inline Status add_avx2(AddParams& params) {
	if (!params.lhs.ptr() || !params.rhs.ptr() || !params.output.ptr() || params.lhs.dtype() != DType::f32 ||
		params.rhs.dtype() != DType::f32 || params.output.dtype() != DType::f32 ||
		params.lhs.shape() != params.rhs.shape() || params.lhs.shape() != params.output.shape()) {
		return Status::invalid_argument;
	}

	const auto* lhs = static_cast<const float*>(params.lhs.ptr());
	const auto* rhs = static_cast<const float*>(params.rhs.ptr());
	auto* output = static_cast<float*>(params.output.ptr());
	const size_t element_num = params.output.numel();

	size_t i = 0;
	for (; i + 8 <= element_num; i += 8) {
		const __m256 left = _mm256_loadu_ps(lhs + i);
		const __m256 right = _mm256_loadu_ps(rhs + i);
		_mm256_storeu_ps(output + i, _mm256_add_ps(left, right));
	}
	for (; i < element_num; ++i) {
		output[i] = lhs[i] + rhs[i];
	}

	return Status::ok;
}

}  // namespace llm_bricks::detail

#else

namespace llm_bricks::detail {

inline Status add_avx2(AddParams&) {
	return Status::unsupported;
}

}  // namespace llm_bricks::detail

#endif
