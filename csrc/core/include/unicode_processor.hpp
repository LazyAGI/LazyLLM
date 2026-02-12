#pragma once

#include <array>
#include <functional>
#include <string_view>
#include <vector>
#include <utf8proc.h>

namespace lazyllm {

class UnicodeProcessor {
public:
    UnicodeProcessor(const std::string_view& text) : _text(text), _text_len(text.size()) {}
    std::vector<std::string_view> split_to_chars() const;
    std::vector<std::string_view> split_by_punctuation() const;

private:
    using Utf8Visitor = std::function<void(size_t offset, size_t byte_len, utf8proc_int32_t codepoint)>;

    void for_each_utf8_unit(const Utf8Visitor& visitor) const;
    static bool is_sentence_punctuation(char32_t codepoint) {
        return std::find(kPunctuationCodepoints.begin(), kPunctuationCodepoints.end(),
            codepoint) != kPunctuationCodepoints.end();
    }

    static const std::array<char32_t, 9> kPunctuationCodepoints;
    std::string_view _text;
    size_t _text_len = 0;
};

} // namespace lazyllm
