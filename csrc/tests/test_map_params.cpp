#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

#include "map_params.hpp"

TEST(MapParams, ExplicitValueHasHighestPriority) {
    lazyllm::MapParams params;
    params.set_default("chunk_size", 1024u);

    const auto value = params.get_param_value<unsigned>("chunk_size", 4096u, 2048u);
    EXPECT_EQ(value, 4096u);
}

TEST(MapParams, ReadsStoredDefaultAndFallback) {
    lazyllm::MapParams params;
    params.set_default("worker_num", 8u);

    EXPECT_EQ(params.get_param_value<unsigned>("worker_num", std::nullopt, 4u), 8u);
    EXPECT_EQ(params.get_param_value<unsigned>("missing", std::nullopt, 4u), 4u);
}

TEST(MapParams, BulkSetAndTypedGetDefault) {
    lazyllm::MapParams params;
    lazyllm::MapParams::MapType updates{
        {"encoding", std::string("gpt2")},
        {"overlap", 128u},
    };
    params.set_default(updates);

    const auto encoding = params.get_default<std::string>("encoding");
    const auto overlap = params.get_default<unsigned>("overlap");
    const auto missing = params.get_default<unsigned>("chunk_size");

    ASSERT_TRUE(encoding.has_value());
    ASSERT_TRUE(overlap.has_value());
    EXPECT_EQ(*encoding, "gpt2");
    EXPECT_EQ(*overlap, 128u);
    EXPECT_FALSE(missing.has_value());
}

TEST(MapParams, ResetDefaultClearsState) {
    lazyllm::MapParams params;
    params.set_default("k", 1);
    ASSERT_FALSE(params.get_default().empty());

    params.reset_default();
    EXPECT_TRUE(params.get_default().empty());
}
