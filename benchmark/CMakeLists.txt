add_executable(bench_ops
	main.cpp
	add.cpp
	gelu.cpp
	layer_norm.cpp
	matmul.cpp
	rope.cpp
	scale.cpp
	softmax.cpp
	gemv.cpp
)
target_link_libraries(bench_ops PRIVATE llm_bricks::llm_bricks)