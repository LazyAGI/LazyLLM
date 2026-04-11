#pragma once

#include <array>
#include <functional>
#include <string_view>
#include <vector>
#include <utf8proc.h>

namespace lazyllm {

class UnicodeProcessor {
public:
    UnicodeProcessor(std::string_view text) : _text(text) {}
    std::vector<std::string_view> split_to_chars() const;
    std::vector<std::string_view> split_by_sentence_endings() const;
    std::vector<std::string_view> split_by_punctuation() const;

private:
    using Utf8Visitor = std::function<void(size_t offset, size_t byte_len, utf8proc_int32_t codepoint)>;

    void for_each_utf8_unit(const Utf8Visitor& visitor) const;
    static bool is_sentence_ending_punctuation(char32_t codepoint) {
        return std::find(kSentenceEndingCodepoints.begin(), kSentenceEndingCodepoints.end(),
            codepoint) != kSentenceEndingCodepoints.end();
    }

    static bool is_sub_sentence_punctuation(char32_t codepoint) {
        return std::find(kSubSentencePunctuationCodepoints.begin(), kSubSentencePunctuationCodepoints.end(),
            codepoint) != kSubSentencePunctuationCodepoints.end();
    }

    static const std::array<char32_t, 6> kSentenceEndingCodepoints;
    static const std::array<char32_t, 10> kSubSentencePunctuationCodepoints;
    std::string_view _text;
};

} // namespace lazyllm
