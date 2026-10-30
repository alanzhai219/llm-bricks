#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

namespace {

void usage() {
    std::cout << "Usage: bench_ops --op add|gelu|layer_norm|matmul|rope|scale|softmax|gemv|token_embd "
                 "[--shape D1,D2,...] [--backend reference|automatic|avx2] "
                 "[--warmup N] [--iters N] [--perf]\n"
                 "  matmul shape: M,K,N or B,M,K,N (A=[M,K], B=[K,N])\n"
                 "  rope shape: B,H,S,D (D must be even)\n"
                 "  other ops: any non-empty shape; last dim is the row width\n";
}

size_t number(const std::string& text) {
    size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (text.empty() || text[0] == '-' || consumed != text.size() || value > std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("invalid size: " + text);
    }
    return static_cast<size_t>(value);
}

std::vector<size_t> parse_shape(const std::string& text) {
    std::vector<size_t> shape;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(',', begin);
        const size_t dim = number(text.substr(begin, end - begin));
        if (dim == 0) throw std::invalid_argument("shape dimensions must be positive");
        shape.push_back(dim);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    size_t count = 1;
    for (size_t dim : shape) {
        if (count > std::numeric_limits<size_t>::max() / dim / sizeof(float)) {
            throw std::invalid_argument("shape is too large");
        }
        count *= dim;
    }
    return shape;
}

bench::Options parse_args(int argc, char** argv) {
    bench::Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            usage();
            std::exit(0);
        }
        if (arg == "--perf") {
            options.perf = true;
            continue;
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--op") options.op = value;
        else if (arg == "--shape") options.shape = parse_shape(value);
        else if (arg == "--warmup") options.warmup = number(value);
        else if (arg == "--iters") options.iters = number(value);
        else if (arg == "--backend") {
            if (value == "reference") options.backend = llm_bricks::Backend::reference;
            else if (value == "automatic") options.backend = llm_bricks::Backend::automatic;
            else if (value == "avx2") options.backend = llm_bricks::Backend::avx2;
            else if (value == "x86_opt") options.backend = llm_bricks::Backend::x86_opt;
            else throw std::invalid_argument("unknown backend: " + value);
        } else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.op.empty()) throw std::invalid_argument("--op is required");
    if (options.iters == 0) throw std::invalid_argument("--iters must be positive");
    if (options.shape.empty()) {
        options.shape = options.op == "rope" ? std::vector<size_t>{1, 8, 128, 64}
                      : options.op == "matmul" ? std::vector<size_t>{128, 128, 128}
                                               : std::vector<size_t>{32, 128};
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bench::Options options = parse_args(argc, argv);
        std::cout << "op=" << options.op << " shape=";
        for (size_t i = 0; i < options.shape.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << options.shape[i];
        }
        std::cout << " warmup=" << options.warmup << " iters=" << options.iters << '\n';

        if (options.op == "add") bench::add(options);
        else if (options.op == "gelu") bench::gelu(options);
        else if (options.op == "layer_norm") bench::layer_norm(options);
        else if (options.op == "matmul") bench::matmul(options);
        else if (options.op == "rope") bench::rope(options);
        else if (options.op == "scale") bench::scale(options);
        else if (options.op == "softmax") bench::softmax(options);
        else if (options.op == "gemv") bench::gemv(options);
        else if (options.op == "token_embd") bench::token_embd(options);
        else throw std::invalid_argument("unknown op: " + options.op);
    } catch (const std::exception& error) {
        std::cerr << "bench_ops: " << error.what() << '\n';
        return 1;
    }
}