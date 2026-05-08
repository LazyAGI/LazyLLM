#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "doc_node.hpp"

TEST(doc_node_core, constructor_sets_text) {
    lazyllm::DocNodeCore node("hello", {}, "fixed-uid");
    EXPECT_EQ(node._text, "hello");
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

    // Sort lines before comparing to avoid relying on unordered_map iteration order.
    auto result_all = node.get_metadata_string(lazyllm::MetadataMode::ALL);
    std::vector<std::string> lines;
    std::istringstream ss(result_all);
    std::string line;
    while (std::getline(ss, line)) lines.push_back(line);
    std::sort(lines.begin(), lines.end());
    EXPECT_EQ(lines, (std::vector<std::string>{"alpha: A", "beta: B"}));
    node._excluded_llm_metadata_keys = {"beta"};
    node._excluded_embed_metadata_keys = {"alpha"};
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::LLM), "alpha: A");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::EMBED), "beta: B");
    EXPECT_EQ(node.get_metadata_string(lazyllm::MetadataMode::NONE), "");
}
