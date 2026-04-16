#include <gtest/gtest.h>

#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.hpp"

TEST(utils, join_lines_returns_empty_for_empty_input) {
    EXPECT_EQ(lazyllm::JoinLines({}), "");
}

TEST(utils, join_lines_uses_newline_separator_by_default) {
    EXPECT_EQ(lazyllm::JoinLines({"a", "b", "c"}), "a\nb\nc");
}

TEST(utils, join_lines_supports_custom_delimiter) {
    EXPECT_EQ(lazyllm::JoinLines({"a", "b", "c"}, ','), "a,b,c");
}

TEST(utils, concat_vector_appends_right_sequence) {
    const auto merged = lazyllm::ConcatVector(std::vector<int>{1, 2}, std::vector<int>{3, 4});
    EXPECT_EQ(merged, (std::vector<int>{1, 2, 3, 4}));
}

TEST(utils, set_union_returns_all_unique_values) {
    const std::set<int> left{1, 2, 3};
    const std::set<int> right{3, 4};
    EXPECT_EQ(lazyllm::SetUnion(left, right), (std::set<int>{1, 2, 3, 4}));
}

TEST(utils, set_diff_returns_only_left_unique_values) {
    const std::set<int> left{1, 2, 3};
    const std::set<int> right{3, 4};
    EXPECT_EQ(lazyllm::SetDiff(left, right), (std::set<int>{1, 2}));
}

TEST(utils, to_hex_returns_lowercase_hex_text) {
    EXPECT_EQ(lazyllm::to_hex(255u), "ff");
}

TEST(utils, generate_uuid_matches_expected_pattern) {
    const std::string uuid = lazyllm::GenerateUUID();
    const std::regex pattern("^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$");
    EXPECT_TRUE(std::regex_match(uuid, pattern));
}

TEST(utils, is_adjacent_returns_true_for_contiguous_views) {
    const std::string text = "abcdef";
    const std::string_view left = std::string_view(text.data(), 3);
    const std::string_view right = std::string_view(text.data() + 3, 3);
    EXPECT_TRUE(lazyllm::is_adjacent(left, right));
}

TEST(utils, is_adjacent_returns_false_for_non_contiguous_views) {
    const std::string text = "abcdef";
    const std::string_view left = std::string_view(text.data(), 3);
    const std::string_view right = std::string_view(text.data() + 4, 2);
    EXPECT_FALSE(lazyllm::is_adjacent(left, right));
}

TEST(utils, chunk_operator_plus_equals_accumulates_fields) {
    lazyllm::Chunk l{"ab", true, 2};
    lazyllm::Chunk r{"cd", false, 3};

    l += r;
    EXPECT_EQ(l.text, "abcd");
    EXPECT_FALSE(l.is_sentence);
    EXPECT_EQ(l.token_size, 5);
}

TEST(utils, rag_metadata_keys_constants_are_exposed) {
    EXPECT_EQ(lazyllm::RAGMetadataKeys::DOC_PATH, "lazyllm_doc_path");
    EXPECT_EQ(lazyllm::RAGMetadataKeys::DOC_ID, "docid");
}

TEST(utils, any_to_string_formats_scalar_metadata_values) {
    EXPECT_EQ(lazyllm::any_to_string(lazyllm::MetadataVType(std::string("alpha"))), "alpha");
    EXPECT_EQ(lazyllm::any_to_string(lazyllm::MetadataVType(7)), "7");
    EXPECT_EQ(lazyllm::any_to_string(lazyllm::MetadataVType(3.5)), "3.5");
}

TEST(utils, any_to_string_formats_vector_metadata_values_with_brackets) {
    EXPECT_EQ(
        lazyllm::any_to_string(lazyllm::MetadataVType(std::vector<std::string>{"a", "b"})),
        "[a,b]");
    EXPECT_EQ(
        lazyllm::any_to_string(lazyllm::MetadataVType(std::vector<int>{1, 2, 3})),
        "[1,2,3]");
    EXPECT_EQ(
        lazyllm::any_to_string(lazyllm::MetadataVType(std::vector<double>{1.5, 2.0})),
        "[1.5,2]");
}

TEST(utils, any_to_string_formats_none_metadata_value) {
    EXPECT_EQ(lazyllm::any_to_string(lazyllm::MetadataVType(std::nullopt)), "None");
}

TEST(utils, any_to_string_formats_string_map_metadata_values) {
    const std::string result = lazyllm::any_to_string(
        lazyllm::MetadataVType(std::unordered_map<std::string, std::string>{
            {"lang", "en"}, {"type", "text"}
        }));
    EXPECT_NE(result.find("lang:en"), std::string::npos);
    EXPECT_NE(result.find("type:text"), std::string::npos);
    EXPECT_EQ(result.front(), '{');
    EXPECT_EQ(result.back(), '}');
}
