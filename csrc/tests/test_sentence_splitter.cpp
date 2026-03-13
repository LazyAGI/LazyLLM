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

TEST(sentence_splitter, merge_chunks_applies_overlap) {
    TestSentenceSplitter splitter(5, 2);

    const std::vector<lazyllm::ChunkView> splits{
        {"ab", false, 2},
        {"cd", false, 2},
        {"ef", false, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 5);
    EXPECT_EQ(merged, (std::vector<std::string>{"abcd", "cdef"}));
}

TEST(sentence_splitter, merge_chunks_throws_on_oversized_single_split) {
    TestSentenceSplitter splitter(3, 1);
    const std::vector<lazyllm::ChunkView> splits{
        {"abcd", false, 4},
    };

    EXPECT_THROW((void)splitter.merge_chunks(splits, 3), std::runtime_error);
}

TEST(sentence_splitter, merge_chunks_shrinks_overlap_to_fit_next_chunk) {
    TestSentenceSplitter splitter(5, 4);

    const std::vector<lazyllm::ChunkView> splits{
        {"aa", false, 2},
        {"b", false, 1},
        {"cccc", false, 4},
        {"dd", false, 2},
        {"ee", false, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 5);
    EXPECT_EQ(merged, (std::vector<std::string>{"aab", "bcccc", "ddee"}));
}
