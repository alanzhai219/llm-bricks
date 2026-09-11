# llm-bricks

`llm-bricks` is a compact C++ operator library for assembling LLM inference.

## Architecture

- **Micro operators** are reusable numerical kernels such as RoPE, matmul,
	softmax, and normalization.
- **Macro operators** compose micro operators into LLM-level work such as
	attention, logits processing, sampling, and decoding.
- Each micro operator has one public API and a readable **reference** backend
	that defines its semantics. Optimized AVX2, AVX-512, CUDA, and SYCL backends
	are added beside that reference implementation.

```text
include/               Stable public C++ API
src/runtime/           Context, dispatch, allocation, and CPU feature support
src/micro/<operator>/  API, reference, and backend-specific implementations
src/macro/             LLM-level operator compositions
tests/                 Reference and backend differential tests
benchmarks/            Per-operator and end-to-end benchmarks
examples/              Minimal integration programs
```

The public call site is independent of the selected backend:

```cpp
#include "micro/rope.hpp"

llm_bricks::Context context;
llm_bricks::rope(context, { ....base = 10000.0F});
```

At present, the scaffold includes a CPU `f32` reference implementation of
in-place RoPE for contiguous `[B, H, S, D]` tensors. Requests for unavailable
backends return `Status::unsupported`; they never silently change semantics.

## Build and test

```text
cmake -S . -B build -DLLMBRICKS_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
