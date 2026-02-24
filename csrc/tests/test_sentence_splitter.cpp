#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sentence_splitter.hpp"
#include "utils.hpp"

namespace {

class TestSentenceSplitter final : public lazyllm::SentenceSplitter {
public:
    TestSentenceSplitter(unsigned chunk_size, unsigned overlap)
        : lazyllm::SentenceSplitter(chunk_size, overlap, 0) {}

    using lazyllm::SentenceSplitter::merge_chunks;
};

} // namespace

TEST(SentenceSplitter, MergeChunksAppliesOverlapAndTrim) {
    TestSentenceSplitter splitter(5, 2);

    const std::vector<lazyllm::ChunkView> splits{
        {"ab", false, 2},
        {"cd", false, 2},
        {"ef ", false, 2},
        {"  ", false, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 5);
    EXPECT_EQ(merged, (std::vector<std::string>{"abcd", "cdef", "ef"}));
}

TEST(SentenceSplitter, MergeChunksThrowsOnOversizedSingleSplit) {
    TestSentenceSplitter splitter(3, 1);
    const std::vector<lazyllm::ChunkView> splits{
        {"abcd", false, 4},
    };

    EXPECT_THROW((void)splitter.merge_chunks(splits, 3), std::runtime_error);
}

TEST(SentenceSplitter, MergeChunksDropsWhitespaceOnlyChunks) {
    TestSentenceSplitter splitter(8, 2);
    const std::vector<lazyllm::ChunkView> splits{
        {"   ", false, 3},
    };

    const auto merged = splitter.merge_chunks(splits, 8);
    EXPECT_TRUE(merged.empty());
}
