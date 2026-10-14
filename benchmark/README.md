# Operator benchmark

## Build

- From the repository root, enable benchmarks and build the `bench_ops` target:

	```sh
	cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DLLMBRICKS_BUILD_BENCHMARKS=ON
	cmake --build build-bench --target bench_ops -j
	```

## Run

- Choose an operator and shape:

	```sh
	./build-bench/benchmark/bench_ops --op rope --shape 1,8,128,64 --warmup 5 --iters 20
	./build-bench/benchmark/bench_ops --op matmul --shape 128,256,64 --backend reference
	./build-bench/benchmark/bench_ops --op softmax --shape 32,512
	```

- Supported operators: `add`, `gelu`, `layer_norm`, `matmul`, `rope`, `scale`, `softmax`.
- `--shape`: positive dimensions separated by commas. Omit it to use the default shape.
	- `matmul`: `M,K,N` or batched `B,M,K,N`.
	- `rope`: `B,H,S,D`, where `D` must be even.
	- Other operators: any non-empty shape; `softmax` and `layer_norm` operate along the last dimension.
- `--backend`: `reference` (default), `automatic`, or `avx2` when supported by the operator.
- `--warmup` / `--iters`: warmup runs (default 5) and measured runs (default 20).
- Results show min, median, and mean in microseconds, plus `output[0]`.
	Input construction and normal input resets are outside the timed interval;
	an unsupported backend or invalid input returns a nonzero exit status.
- For more repeatable single-thread measurements, set `OMP_NUM_THREADS=1` when applicable.

## Optional Linux perf trace

- Enable profiling with `--perf` **and** JSON dumping with `LINUX_PERF=dump` or `LINUX_PERF=dump=N`:

	```sh
	LINUX_PERF=dump=1 ./build-bench/benchmark/bench_ops --op rope --shape 1,8,128,64 --perf
	```

- `--perf` profiles each measured iteration and one additional invocation using `linux_perf.hpp`.
	With `dump=1`, only the first completed profile is recorded; increase `N` to record more.
- The output file `perf_dump.json` is created in the **current working directory**,
	not necessarily beside the executable. Without `LINUX_PERF=dump`, no JSON trace is written.
- Perf counters require Linux perf-event permissions (for example, an appropriate
	`perf_event_paranoid` setting). Perf counts are not printed to stdout.
- The reported wall-clock timings with `--perf` include profiling overhead and an
	additional input reset inside the timed interval. Use runs without `--perf`
	to compare operator latency.

## Layout

- Each operator has its own `.cpp` file in this directory.
- `main.cpp` parses arguments and selects the operator; `common.hpp` implements timing and optional profiling.