#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace test {

inline bool check(bool condition, const char* expression, const char* file, int line) {
    if (condition) return true;
    std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
    return false;
}

inline bool check_near(float actual, float expected, float tolerance, const char* file, int line) {
    if (std::fabs(actual - expected) <= tolerance) return true;
    std::cerr << file << ':' << line << ": expected " << expected << ", got " << actual << '\n';
    return false;
}

} // namespace test

#define CHECK(condition) if (!test::check((condition), #condition, __FILE__, __LINE__)) return EXIT_FAILURE
#define CHECK_NEAR(actual, expected, tolerance) if (!test::check_near((actual), (expected), (tolerance), __FILE__, __LINE__)) return EXIT_FAILURE