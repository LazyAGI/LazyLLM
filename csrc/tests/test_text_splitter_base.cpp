#include <gtest/gtest.h>

#include <memory>
#include <optional>
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

class TestTextSplitter final : public lazyllm::TextSplitterBase {
public:
    TestTextSplitter(unsigned chunk_size, unsigned overlap)
        : lazyllm::TextSplitterBase(chunk_size, overlap, 0) {}

    using lazyllm::TextSplitterBase::merge_chunks;
    using lazyllm::TextSplitterBase::split_recursive;
};

} // namespace

TEST(TextSplitterBase, ConstructorValidatesParameters) {
    EXPECT_THROW((void)TestTextSplitter(10, 11), std::runtime_error);
    EXPECT_THROW((void)TestTextSplitter(0, 0), std::runtime_error);
}

TEST(TextSplitterBase, SplitTextKeepingSeparator) {
    const auto parts = lazyllm::TextSplitterBase::split_text_while_keeping_separator("a--b--", "--");
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0], "a--");
    EXPECT_EQ(parts[1], "b--");

    const auto leading_sep = lazyllm::TextSplitterBase::split_text_while_keeping_separator("--x", "--");
    ASSERT_EQ(leading_sep.size(), 1u);
    EXPECT_EQ(leading_sep[0], "x");
}

TEST(TextSplitterBase, SplitTextChecksMetadataBudget) {
    TestTextSplitter splitter(60, 0);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());

    EXPECT_THROW((void)splitter.split_text("abc", 60), std::runtime_error);
    EXPECT_THROW((void)splitter.split_text("abc", 11), std::runtime_error);
}

TEST(TextSplitterBase, SplitRecursiveFallsBackToCharLevel) {
    TestTextSplitter splitter(100, 0);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());

    const auto chunks = splitter.split_recursive("abc", 2);
    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(chunks[0].view, "a");
    EXPECT_EQ(chunks[1].view, "b");
    EXPECT_EQ(chunks[2].view, "c");
    EXPECT_FALSE(chunks[0].is_sentence);
}

TEST(TextSplitterBase, MergeChunksUsesOverlap) {
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

TEST(TextSplitterBase, TransformReturnsChunkNodesAndSupportsNull) {
    TestTextSplitter splitter(100, 0);
    splitter.set_tokenizer(std::make_shared<ByteTokenizer>());

    auto global_meta = std::make_shared<lazyllm::DocNode::Metadata>();
    lazyllm::DocNode node("hello", "", "", nullptr, {}, global_meta);

    const auto chunks = splitter.transform(&node);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].get_text(lazyllm::MetadataMode::NONE), "hello");

    const auto empty = splitter.transform(nullptr);
    EXPECT_TRUE(empty.empty());
}

TEST(TextSplitterBase, FromTiktokenEncoderThrowsOnInvalidName) {
    TestTextSplitter splitter(100, 0);
    EXPECT_THROW((void)splitter.from_tiktoken_encoder("definitely_unknown"), std::runtime_error);
}
