cmake_minimum_required(VERSION 3.20)

project(llm_bricks VERSION 0.1.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Use Release by default.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
else()
    add_compile_options(-O2)
endif()

option(LLMBRICKS_INSTALL "Enable installation" ON)
option(LLMBRICKS_BUILD_TESTS "Build tests" ON)
option(LLMBRICKS_BUILD_BENCHMARKS "Build operator benchmarks" OFF)
set(LLMBRICKS_PARALLEL_BACKENDS serial omp threadpool)
set(LLMBRICKS_PARALLEL_BACKEND "serial" CACHE STRING "Parallel backend: serial, omp, or threadpool")
set_property(CACHE LLMBRICKS_PARALLEL_BACKEND PROPERTY STRINGS serial omp threadpool)

if(NOT LLMBRICKS_PARALLEL_BACKEND IN_LIST LLMBRICKS_PARALLEL_BACKENDS)
    message(FATAL_ERROR "LLMBRICKS_PARALLEL_BACKEND must be serial, omp, or threadpool")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
endif()

add_library(llm_bricks STATIC
    src/runtime/context.cpp
    src/micro/add/add.cpp
    src/micro/gelu/gelu.cpp
    src/micro/layer_norm/layer_norm.cpp
    src/micro/matmul/matmul.cpp
    src/micro/rope/rope.cpp
    src/micro/scale/scale.cpp
    src/micro/softmax/softmax.cpp
    src/micro/gemv/gemv.cpp
)
add_library(llm_bricks::llm_bricks ALIAS llm_bricks)

target_include_directories(llm_bricks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(llm_bricks PUBLIC cxx_std_17)
target_compile_options(llm_bricks PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma)
if(LLMBRICKS_PARALLEL_BACKEND STREQUAL "threadpool")
    target_compile_definitions(llm_bricks PRIVATE LLMBRICKS_THREAD_PARALLEL)
    target_link_libraries(llm_bricks PRIVATE -lpthread)
elseif(LLMBRICKS_PARALLEL_BACKEND STREQUAL "omp")
    target_compile_definitions(llm_bricks PRIVATE LLMBRICKS_OMP_PARALLEL)
    target_link_libraries(llm_bricks PUBLIC OpenMP::OpenMP_CXX)
endif()

if(LLMBRICKS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(LLMBRICKS_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(LLMBRICKS_INSTALL)
    install(TARGETS llm_bricks
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY include/ DESTINATION include)
endif()
