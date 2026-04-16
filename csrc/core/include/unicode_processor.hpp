#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace lazyllm {

class UnicodeProcessor {
public:
    explicit UnicodeProcessor(std::string_view text) : _text(text) {}
    UnicodeProcessor(std::string&&) = delete;
    UnicodeProcessor(const std::string&&) = delete;

    std::vector<std::string_view> split_to_chars() const;
    std::vector<std::string_view> split_by_sentence_endings() const;
    std::vector<std::string_view> split_by_punctuation() const;
    std::vector<std::string_view> split_to_n_parts(size_t n) const;

private:
    template <typename Visitor>
    void for_each_utf8_unit(Visitor&& visitor) const;
    static bool is_sentence_ending_punctuation(char32_t codepoint) {
        return std::binary_search(kSentenceEndingCodepoints.begin(), kSentenceEndingCodepoints.end(), codepoint);
    }

    static bool is_sub_sentence_punctuation(char32_t codepoint) {
        return std::binary_search(kSubSentencePunctuationCodepoints.begin(), kSubSentencePunctuationCodepoints.end(),
            codepoint);
    }

    static const std::array<char32_t, 6> kSentenceEndingCodepoints;
    static const std::array<char32_t, 10> kSubSentencePunctuationCodepoints;
    std::string_view _text;
};

} // namespace lazyllm
