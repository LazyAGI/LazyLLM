#include <gtest/gtest.h>

#include <regex>
#include <set>
#include <string>
#include <vector>

#include "utils.hpp"

TEST(Utils, JoinLinesAndConcatVector) {
    EXPECT_EQ(lazyllm::JoinLines({}), "");
    EXPECT_EQ(lazyllm::JoinLines({"a", "b", "c"}), "a\nb\nc");
    EXPECT_EQ(lazyllm::JoinLines({"a", "b", "c"}, ','), "a,b,c");

    const auto merged = lazyllm::ConcatVector(std::vector<int>{1, 2}, std::vector<int>{3, 4});
    EXPECT_EQ(merged, (std::vector<int>{1, 2, 3, 4}));
}

TEST(Utils, SetUnionAndSetDiff) {
    const std::set<int> left{1, 2, 3};
    const std::set<int> right{3, 4};

    EXPECT_EQ(lazyllm::SetUnion(left, right), (std::set<int>{1, 2, 3, 4}));
    EXPECT_EQ(lazyllm::SetDiff(left, right), (std::set<int>{1, 2}));
}

TEST(Utils, HexUuidAndAdjacency) {
    EXPECT_EQ(lazyllm::to_hex(255u), "ff");

    const std::string uuid = lazyllm::GenerateUUID();
    const std::regex pattern("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$");
    EXPECT_TRUE(std::regex_match(uuid, pattern));

    const std::string text = "abcdef";
    const std::string_view left = std::string_view(text.data(), 3);
    const std::string_view right_adjacent = std::string_view(text.data() + 3, 3);
    const std::string_view right_not_adjacent = std::string_view(text.data() + 4, 2);
    EXPECT_TRUE(lazyllm::is_adjacent(left, right_adjacent));
    EXPECT_FALSE(lazyllm::is_adjacent(left, right_not_adjacent));
}

TEST(Utils, ChunkOperatorAccumulatesFields) {
    lazyllm::Chunk l{"ab", true, 2};
    lazyllm::Chunk r{"cd", false, 3};

    l += r;
    EXPECT_EQ(l.text, "abcd");
    EXPECT_FALSE(l.is_sentence);
    EXPECT_EQ(l.token_size, 5);
}

TEST(Utils, MetadataKeyConstantsExposed) {
    EXPECT_EQ(lazyllm::RAGMetadataKeys::DOC_PATH, "lazyllm_doc_path");
    EXPECT_EQ(lazyllm::RAGMetadataKeys::DOC_ID, "docid");
}
