#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "unicode_processor.hpp"

TEST(unicode_processor, split_to_chars_supports_multibyte) {
    const std::string text = "a你🙂";
    const lazyllm::UnicodeProcessor processor(text);

    const auto chars = processor.split_to_chars();
    EXPECT_EQ(chars, (std::vector<std::string_view>{"a", "你", "🙂"}));
}

TEST(unicode_processor, split_by_punctuation_for_ascii) {
    const std::string text = "Hello,world!";
    const lazyllm::UnicodeProcessor processor(text);

    const auto chunks = processor.split_by_punctuation();
    EXPECT_EQ(chunks, (std::vector<std::string_view>{"Hello,", "world!"}));
}

TEST(unicode_processor, split_by_punctuation_for_cjk) {
    const std::string text = "你好。世界！";
    const lazyllm::UnicodeProcessor processor(text);

    const auto chunks = processor.split_by_punctuation();
    EXPECT_EQ(chunks, (std::vector<std::string_view>{"你好。", "世界！"}));
}
