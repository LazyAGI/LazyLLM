#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "unicode_processor.hpp"

namespace {

std::vector<std::string> ToStrings(const std::vector<std::string_view>& views) {
    std::vector<std::string> out;
    out.reserve(views.size());
    for (const auto& view : views) out.emplace_back(view);
    return out;
}

} // namespace

TEST(UnicodeProcessor, SplitToCharsSupportsMultibyte) {
    const std::string text = "a你🙂";
    const lazyllm::UnicodeProcessor processor(text);

    const auto chars = ToStrings(processor.split_to_chars());
    EXPECT_EQ(chars, (std::vector<std::string>{"a", "你", "🙂"}));
}

TEST(UnicodeProcessor, SplitByPunctuationHandlesAsciiAndCjk) {
    const std::string text = "Hello,world。你好！";
    const lazyllm::UnicodeProcessor processor(text);

    const auto chunks = ToStrings(processor.split_by_punctuation());
    EXPECT_EQ(chunks, (std::vector<std::string>{"Hello,", "world。", "你好！"}));
}

TEST(UnicodeProcessor, SplitByPunctuationFallbackWhenOnlyPunctuation) {
    const std::string text = "!!!";
    const lazyllm::UnicodeProcessor processor(text);

    const auto chunks = ToStrings(processor.split_by_punctuation());
    EXPECT_EQ(chunks, (std::vector<std::string>{"!!!"}));
}
