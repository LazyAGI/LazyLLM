#include <gtest/gtest.h>

#include "lazyllm.hpp"

TEST(LazyLLM, Smoke) {
    EXPECT_GT(PYBIND11_VERSION_MAJOR, 0);
}
