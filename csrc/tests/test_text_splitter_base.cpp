#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "doc_node.hpp"
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
            out.push_back(static_cast<char>(token_id));
        }
        return out;
    }
};

class TestTextSplitter final : public lazyllm::TextSplitterBase {
public:
    TestTextSplitter(unsigned chunk_size, unsigned overlap = 0)
        : lazyllm::TextSplitterBase(chunk_size, overlap, 0) {}

    using lazyllm::TextSplitterBase::merge_chunks;
    using lazyllm::TextSplitterBase::split_recursive;
};

} // namespace

TEST(text_splitter_base, exception_management) {
    EXPECT_THROW((void)lazyllm::TextSplitterBase(10, 11), std::runtime_error);
    EXPECT_THROW((void)lazyllm::TextSplitterBase(0, 0), std::runtime_error);
}

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
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());
    EXPECT_THROW((void)splitter.split_text("abc", 60), std::runtime_error);
}

TEST(text_splitter_base, split_text_throws_when_metadata_budget_too_small) {
    lazyllm::TextSplitterBase splitter(60, 0);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());
    EXPECT_THROW((void)splitter.split_text("abc", 11), std::runtime_error);
}

TEST(text_splitter_base, split_recursive_falls_back_to_char_level) {
    TestTextSplitter splitter(100, 0);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());

    const auto chunks = splitter.split_recursive("abc", 2);
    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0], "a");
    EXPECT_EQ(chunks[1], "b");
    EXPECT_EQ(chunks[2], "c");
    EXPECT_FALSE(chunks[0].is_sentence);
}

TEST(text_splitter_base, merge_chunks_uses_overlap) {
    TestTextSplitter splitter(100, 1);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());

    const std::vector<lazyllm::ChunkView> splits{
        {"ab", true, 2},
        {"cd", true, 2},
        {"ef", true, 2},
    };

    const auto merged = splitter.merge_chunks(splits, 4);
    EXPECT_EQ(merged, (std::vector<std::string>{"ab", "bcd", "def"}));
}

TEST(text_splitter_base, transform_returns_chunk_nodes) {
    lazyllm::TextSplitterBase splitter(100, 0);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());

    lazyllm::PDocNode node = std::make_shared<lazyllm::DocNode>("hello");

    auto chunks = splitter.transform(node);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0]->get_text(), "hello");

    chunks = splitter.transform(nullptr);
    EXPECT_TRUE(chunks.empty());
}

TEST(text_splitter_base, from_tiktoken_encoder_throws_on_invalid_name) {
    lazyllm::TextSplitterBase splitter(100, 0);
    EXPECT_THROW((void)splitter.from_tiktoken_encoder("definitely_unknown"), std::runtime_error);
}
