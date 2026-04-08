#include <gtest/gtest.h>

#include <string>

#include "doc_node.hpp"

TEST(doc_node_core, constructor_sets_text) {
    lazyllm::DocNodeCore node("hello", "fixed-uid");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "hello");
    EXPECT_EQ(node.get_text_view(), "hello");
}

TEST(doc_node_core, set_text_view_replaces_text) {
    lazyllm::DocNodeCore node("hello");
    node.set_text_view("world");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "world");
    EXPECT_EQ(node.get_text_view(), "world");
}

TEST(doc_node_core, metadata_string_and_text) {
    lazyllm::DocNodeCore node("body");
    node._metadata = lazyllm::DocNodeCore::Metadata{
        {"alpha", std::string("A")},
        {"beta", std::string("B")},
    };

    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::ALL), "alpha: A\nbeta: B");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::NONE), "");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "body");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::ALL), "alpha: A\nbeta: B\n\nbody");
}

