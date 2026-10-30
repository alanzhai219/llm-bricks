#include "common.hpp"
#include "macro/token_embd.hpp"

namespace bench {
void token_embd(const Options& options) {
    using namespace llm_bricks;
    Context context(DeviceType::cpu, options.backend);
    size_t vocab_size = options.shape[0];
    size_t N = options.shape[1];
    size_t input_length = options.shape[2];

    Tensor embedding_table({vocab_size, N, input_length});
    fill(embedding_table);
    std::vector<int> input_ids(input_length);
    Tensor output({input_length, N});
    TokenEmbeddingParams params(embedding_table, input_ids, output);
    TokenEmbedding op(context);
    op.set_input(params);
    measure(options, output, []{}, [&] { return op.infer(); });
}
} // namespace bench