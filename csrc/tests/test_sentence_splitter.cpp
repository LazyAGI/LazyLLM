#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "sentence_splitter.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"

namespace {

class ByteTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(const std::string_view& view) const override {
        std::vector<int> out;
        out.reserve(view.size());
        for (unsigned char ch : view) out.push_back(static_cast<int>(ch));
        return out;
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        std::string out;
        out.reserve(token_ids.size());
        for (int token_id : token_ids) out.push_back(static_cast<char>(token_id));
        return out;
    }
};

class TestSentenceSplitter final : public lazyllm::SentenceSplitter {
public:
    TestSentenceSplitter(unsigned chunk_size, unsigned overlap)
        : lazyllm::SentenceSplitter(chunk_size, overlap, "gpt2") {}

    using lazyllm::SentenceSplitter::merge_chunks;

    void set_tokenizer_for_test(std::shared_ptr<Tokenizer> tokenizer) {
        _tokenizer = std::move(tokenizer);
    }
};

} // namespace

TEST(sentence_splitter, merge_chunks_applies_overlap) {
    TestSentenceSplitter splitter(5, 2);

    const std::vector<lazyllm::Chunk> splits{
        {"ab", false, 2},
        {"cd", false, 2},
        {"ef", false, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 5);
    EXPECT_EQ(merged, (std::vector<std::string>{"abcd", "cdef"}));
}

TEST(sentence_splitter, merge_chunks_throws_on_oversized_single_split) {
    TestSentenceSplitter splitter(3, 1);
    const std::vector<lazyllm::Chunk> splits{
        {"abcd", false, 4},
    };

    EXPECT_THROW((void)splitter.merge_chunks(splits, 3), std::runtime_error);
}

TEST(sentence_splitter, merge_chunks_shrinks_overlap_to_fit_next_chunk) {
    TestSentenceSplitter splitter(5, 4);

    const std::vector<lazyllm::Chunk> splits{
        {"aa", false, 2},
        {"b", false, 1},
        {"cccc", false, 4},
        {"dd", false, 2},
        {"ee", false, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 5);
    EXPECT_EQ(merged, (std::vector<std::string>{"aab", "bcccc", "ddee"}));
}

TEST(sentence_splitter, split_text_empty_input_returns_single_empty_chunk) {
    lazyllm::SentenceSplitter splitter(100, 10, "gpt2");
    const auto chunks = splitter.split_text("", 0);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0], "");
}

TEST(sentence_splitter, split_text_splits_large_text_with_byte_tokenizer) {
    TestSentenceSplitter splitter(60, 0);
    splitter.set_tokenizer_for_test(std::make_shared<ByteTokenizer>());

    std::string text(130, 'x');
    const auto chunks = splitter.split_text(text, 0);
    ASSERT_FALSE(chunks.empty());
    for (const auto& chunk : chunks) EXPECT_LE(chunk.size(), 60u);
}
