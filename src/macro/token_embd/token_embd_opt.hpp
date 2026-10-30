#pragma once

#include "macro/token_embd.hpp"

namespace llm_bricks::detail {

Status token_embd_opt(TokenEmbeddingParams& params) {
    size_t N = params.embedding_table.dim(1);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < params.input_ids.size(); ++i) {
        int idx = params.input_ids[i];
        if (idx < 0 || static_cast<size_t>(idx) >= params.embedding_table.dim(0)) {
            return Status::invalid_argument;
        }
        const float* embedding_row = static_cast<const float*>(params.embedding_table.ptr()) + idx * N;
        float* output_row = static_cast<float*>(params.output.ptr()) + i * N;
        std::memcpy(output_row, embedding_row, N * sizeof(float));
    }

    return Status::ok;
}

} // namespace llm_bricks::detail