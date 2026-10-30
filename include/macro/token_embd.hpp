#pragma once

#include <vector>

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {
struct TokenEmbeddingParams : public BaseParams {
    TokenEmbeddingParams(const Tensor& embedding_table,
                         const std::vector<int>& input_ids,
                         Tensor& output)
        : embedding_table(embedding_table),
          input_ids(input_ids),
          output(output) {}
    const Tensor& embedding_table;
    const std::vector<int>& input_ids;
    Tensor& output;
};

class TokenEmbedding : public BaseOp {
public:
    explicit TokenEmbedding(Context& context) : m_ctx(context) {}
    void set_input(TokenEmbeddingParams& params) { m_params = &params; }
    Status infer() override;
private:
    Context m_ctx;
    TokenEmbeddingParams* m_params = nullptr;
};

} // namespace llm_bricks