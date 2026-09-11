cmake_minimum_required(VERSION 3.20)

set(LLMBRICKS_CXX_COMPILER "g++" CACHE STRING "C++ compiler")
set(CMAKE_CXX_COMPILER "${LLMBRICKS_CXX_COMPILER}")

project(llm_bricks VERSION 0.1.0 LANGUAGES CXX)

# Use Release by default.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LLMBRICKS_INSTALL "Enable installation" ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
endif()

add_library(llm_bricks SHARED
    src/runtime/context.cpp
    src/micro/rope/rope.cpp
)
add_library(llm_bricks::llm_bricks ALIAS llm_bricks)

target_include_directories(llm_bricks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(llm_bricks PUBLIC cxx_std_17)
target_compile_options(llm_bricks PRIVATE -Wall -Wextra -Wpedantic)

if(LLMBRICKS_INSTALL)
    install(TARGETS llm_bricks
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY include/ DESTINATION include)
endif()
