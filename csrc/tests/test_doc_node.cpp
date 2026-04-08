#include <gtest/gtest.h>

#include <string>

#include "doc_node.hpp"

TEST(doc_node_core, constructor_sets_text) {
    lazyllm::DocNodeCore node("hello", {}, "fixed-uid");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "hello");
    EXPECT_EQ(node._uid, "fixed-uid");
}

TEST(doc_node_core, constructor_generates_uid_when_empty) {
    lazyllm::DocNodeCore node("hello");
    EXPECT_FALSE(node._uid.empty());
}

TEST(doc_node_core, metadata_string_and_text) {
    lazyllm::DocNodeCore node("body");
    node._metadata = lazyllm::DocNodeCore::Metadata{
        {"alpha", std::string("A")},
        {"beta", std::string("B")},
    };

    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::ALL), "alpha: A\nbeta: B");
    node._excluded_llm_metadata_keys = {"beta"};
    node._excluded_embed_metadata_keys = {"alpha"};
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::LLM), "alpha: A");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::EMBED), "beta: B");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::NONE), "");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::NONE), "body");
    EXPECT_EQ(node.get_text(lazyllm::MetadataMode::ALL), "alpha: A\nbeta: B\n\nbody");
}
