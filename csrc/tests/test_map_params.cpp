#include <gtest/gtest.h>

#include <any>
#include <string>
#include <unordered_map>

#include "map_params.hpp"

TEST(map_params, get_param_value) {
    lazyllm::MapParams params;
    params.set_default("chunk_size", 1024u);

    auto value = params.get_param_value<unsigned>("chunk_size", 4096u, 2048u);
    EXPECT_EQ(value, 4096u);

    value = params.get_param_value<unsigned>("chunk_size", std::nullopt, 4u);
    EXPECT_EQ(value, 1024u);

    EXPECT_EQ(params.get_param_value<unsigned>("missing", std::nullopt, 4u), 4u);
}

TEST(map_params, bulk_set_updates_defaults) {
    lazyllm::MapParams params;
    lazyllm::MapParams::MapType updates{
        {"encoding", std::string("gpt2")},
        {"overlap", 128u},
    };
    params.set_default(updates);
    EXPECT_EQ(std::any_cast<std::string>(params.get_default().at("encoding")), "gpt2");
    EXPECT_EQ(std::any_cast<unsigned>(params.get_default().at("overlap")), 128u);
}

TEST(map_params, typed_get_default_returns_value) {
    lazyllm::MapParams params;
    params.set_default("encoding", std::string("gpt2"));
    params.set_default("overlap", 128u);

    const auto encoding = params.get_default<std::string>("encoding");
    const auto overlap = params.get_default<unsigned>("overlap");

    ASSERT_TRUE(encoding.has_value());
    ASSERT_TRUE(overlap.has_value());
    EXPECT_EQ(*encoding, "gpt2");
    EXPECT_EQ(*overlap, 128u);
}

TEST(map_params, typed_get_default_returns_nullopt_when_missing) {
    lazyllm::MapParams params;
    const auto missing = params.get_default<unsigned>("chunk_size");
    EXPECT_FALSE(missing.has_value());
}

TEST(map_params, reset_default_clears_state) {
    lazyllm::MapParams params;
    params.set_default("k", 1);
    ASSERT_FALSE(params.get_default().empty());

    params.reset_default();
    EXPECT_TRUE(params.get_default().empty());
}
