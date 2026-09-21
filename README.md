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

The public call site is independent of the selected backend:

```cpp
#include "micro/rope.hpp"

// step1: create context
llm_bricks::Context context;

// step2: create op instance
auto rope_obj = llm_bricks::rope(context);

// step3: prepare op params
llm_bricks::RopeParams params(...);
rope_obj.set_params(*params);

// step4: infer
repo_obj.infer();
```

At present, the scaffold includes a CPU `f32` reference implementation of
in-place RoPE for contiguous `[B, H, S, D]` tensors. Requests for unavailable
backends return `Status::unsupported`; they never silently change semantics.

## Build and test

```text
mkdir build
cd build
cmake ..
make -j
```
