#pragma once

#include "macro/token_embd.hpp"

namespace llm_bricks::detail {

Status token_embd_ref(TokenEmbeddingParams& params) {
    size_t N = params.embedding_table.dim(1);
    for (size_t i = 0; i < params.input_ids.size(); ++i) {
        int idx = params.input_ids[i];
        if (idx < 0 || static_cast<size_t>(idx) >= params.embedding_table.dim(0)) {
            return Status::invalid_argument;
        }
        auto id = static_cast<size_t>(idx);
        for (size_t j = 0; j < N; ++j) {
            float* output_row = static_cast<float*>(params.output.ptr()) + i * N;
            const float* embedding_row = static_cast<const float*>(params.embedding_table.ptr()) + id * N;
            output_row[j] = embedding_row[j];
        }
        // const float* embedding_row = static_cast<const float*>(params.embedding_table.ptr()) + idx * N;
        // float* output_row = static_cast<float*>(params.output.ptr()) + i * N;
        // std::memcpy(output_row, embedding_row, N * sizeof(float));
    }

    return Status::ok;
}

} // namespace llm_bricks::detail