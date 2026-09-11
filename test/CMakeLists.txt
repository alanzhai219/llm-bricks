add_executable(test_rope micro/test_rope.cpp)
target_link_libraries(test_rope PRIVATE llm_bricks)
target_include_directories(test_rope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME rope COMMAND test_rope)