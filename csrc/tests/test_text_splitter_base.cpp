#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "text_splitter_base.hpp"
#include "utils.hpp"

namespace {

class ByteTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(const std::string_view& view) const override {
        std::vector<int> out;
        out.reserve(view.size());
        for (unsigned char ch : view) {
            out.push_back(static_cast<int>(ch));
        }
        return out;
    }

    std::string decode(const std::vector<int>& token_ids) const override {
        std::string out;
        out.reserve(token_ids.size());
        for (int token_id : token_ids) {
            out.push_back(static_cast<char>(static_cast<unsigned char>(token_id)));
        }
        return out;
    }
};

class TestTextSplitter final : public lazyllm::TextSplitterBase {
public:
    TestTextSplitter(unsigned chunk_size, unsigned overlap = 0)
        : lazyllm::TextSplitterBase(chunk_size, overlap, "gpt2") {}

    using lazyllm::TextSplitterBase::merge_chunks;
    using lazyllm::TextSplitterBase::split_recursive;

    void set_tokenizer_for_test(std::shared_ptr<Tokenizer> tokenizer) {
        _tokenizer = std::move(tokenizer);
    }
};

} // namespace

TEST(text_splitter_base, split_text_keep_separator_returns_segments) {
    const auto parts = lazyllm::TextSplitterBase::split_text_while_keeping_separator("a--b--", "--");
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0], "a--");
    EXPECT_EQ(parts[1], "b--");
}

TEST(text_splitter_base, split_text_keep_separator_skips_leading_separator) {
    const auto parts = lazyllm::TextSplitterBase::split_text_while_keeping_separator("--x", "--");
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], "x");
}

TEST(text_splitter_base, split_text_throws_when_metadata_exceeds_chunk_size) {
    lazyllm::TextSplitterBase splitter(60, 0);
    EXPECT_THROW((void)splitter.split_text("abc", 60), std::invalid_argument);
}

TEST(text_splitter_base, split_text_allows_small_metadata_budget) {
    lazyllm::TextSplitterBase splitter(60, 0);
    EXPECT_NO_THROW((void)splitter.split_text("abc", 11));
}

TEST(text_splitter_base, split_recursive_falls_back_to_char_level) {
    TestTextSplitter splitter(100, 0);
    splitter.set_tokenizer_for_test(std::make_shared<ByteTokenizer>());

    const auto chunks = splitter.split_recursive("abc", 2);
    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0].view, "a");
    EXPECT_EQ(chunks[1].view, "b");
    EXPECT_EQ(chunks[2].view, "c");
    EXPECT_FALSE(chunks[0].is_sentence);
}

TEST(text_splitter_base, merge_chunks_uses_overlap) {
    TestTextSplitter splitter(100, 1);
    splitter.set_tokenizer_for_test(std::make_shared<ByteTokenizer>());

    const std::vector<lazyllm::Chunk> splits{
        {"ab", true, 2},
        {"cd", true, 2},
        {"ef", true, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 4);
    EXPECT_EQ(merged, (std::vector<std::string>{"ab", "bcd", "def"}));
}

TEST(text_splitter_base, split_text_returns_single_empty_chunk_for_empty_input) {
    lazyllm::TextSplitterBase splitter(100, 0);
    const auto chunks = splitter.split_text("", 0);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0], "");
}

TEST(text_splitter_base, split_text_uses_current_definition_for_large_inputs) {
    TestTextSplitter splitter(60, 0);
    splitter.set_tokenizer_for_test(std::make_shared<ByteTokenizer>());

    std::string text(120, 'a');
    const auto chunks = splitter.split_text(text, 0);

    ASSERT_FALSE(chunks.empty());
    for (const auto& chunk : chunks) EXPECT_LE(chunk.size(), 60u);
}

TEST(text_splitter_base, merge_chunks_throws_on_oversized_end_split) {
    TestTextSplitter splitter(100, 1);
    splitter.set_tokenizer_for_test(std::make_shared<ByteTokenizer>());

    const std::vector<lazyllm::Chunk> splits{
        {"a", true, 1},
        {"bbbb", true, 4},
    };

    EXPECT_THROW((void)splitter.merge_chunks(splits, 3), std::runtime_error);
}
