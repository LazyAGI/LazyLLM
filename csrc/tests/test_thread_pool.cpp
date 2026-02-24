#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "thread_pool.hpp"

TEST(ThreadPool, ExecutesTasksAndReturnsValues) {
    ThreadPool pool(3);

    auto f1 = pool.enqueue([] { return 1 + 2; });
    auto f2 = pool.enqueue([](int v) { return v * 2; }, 5);

    EXPECT_EQ(f1.get(), 3);
    EXPECT_EQ(f2.get(), 10);
}

TEST(ThreadPool, PropagatesTaskExceptionThroughFuture) {
    ThreadPool pool(1);
    auto failing = pool.enqueue([]() -> int {
        throw std::runtime_error("boom");
    });

    EXPECT_THROW((void)failing.get(), std::runtime_error);
}
