#pragma once

#include <stdlib.h>
#include <iostream>

#define ASSERT_EQ(x,y)                                                                  \
    do {                                                                                \
        if ((x) != (y)) {                                                               \
            std::cerr << "FAILURE: Expected " << (y) << " instead got " << (x)          \
                << " for " << #x << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1;                                                                   \
            }                                                                           \
    } while(0);

#define ASSERT(x) ASSERT_EQ((bool)(x), true)

#define TEST(f, ...)                                             \
    do {                                                         \
        int pass_ = f(__VA_ARGS__);                              \
        if (pass_ != 0) {                                        \
                std::cerr << "TEST FAILED: " << #f << std::endl; \
            } else {                                             \
                std::cout << "TEST PASSED: " << #f << std::endl; \
            }                                                    \
            pass |= pass_;                                       \
    } while(0);

#define TEST_WARGS(fn, ...)                              \
    do {                                                 \
        unsigned pass_ = fn(__VA_ARGS__);                \
        if (pass_ != 0) {                                \
            fprintf(stderr, "TEST FAILED: %s\n", #fn);   \
        } else {                                         \
            fprintf(stdout, "TEST PASSED: %s\n", #fn);   \
        }                                                \
            pass |= pass_;                               \
    } while (0);

#define ASSERT_CLOSE(x,y)                                                           \
    do {                                                                            \
        if(fabs(log2(x) - log2(y)) > (1e-4)) {                                      \
            fprintf(stderr, "TEST FAILURE: Expected: %f got: %f, for: %s %s:%d\n",  \
                y, x, #x, __FILE__, __LINE__);                                      \
            return 1;                                                               \
        }                                                                           \
    } while(0);
