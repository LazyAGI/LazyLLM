#include "unicode_processor.hpp"
namespace lazyllm {

const std::array<char32_t, 9> UnicodeProcessor::kPunctuationCodepoints = {
    U',',
    U'.',
    U';',
    U'!',
    U'\uFF0C', // ，
    U'\uFF1B', // ；
    U'\u3002', // 。
    U'\uFF1F', // ？
    U'\uFF01', // ！
};

void UnicodeProcessor::for_each_utf8_unit(const Utf8Visitor& visitor) const {
    size_t i = 0;
    while (i < _text_len) {
        utf8proc_int32_t codepoint = -1;
        const utf8proc_ssize_t n = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t*>(_text.data() + i),
            static_cast<utf8proc_ssize_t>(_text_len - i),
            &codepoint);

        if (n <= 0) {
            i += 1;
            continue;
        }

        visitor(i, static_cast<size_t>(n), codepoint);
        i += static_cast<size_t>(n);
    }
}

std::vector<std::string_view> UnicodeProcessor::split_to_chars() const {
    std::vector<std::string_view> out;
    if (_text.empty()) return out;
    out.reserve(_text_len); // Grapheme count <= byte length

    size_t cluster_start = std::string_view::npos;
    utf8proc_int32_t prev = -1;
    utf8proc_int32_t state = 0;

    for_each_utf8_unit([&](size_t offset, size_t byte_len, utf8proc_int32_t codepoint) {
        if (cluster_start == std::string_view::npos) {
            cluster_start = offset;
        } else if (utf8proc_grapheme_break_stateful(prev, codepoint, &state)) {
            out.emplace_back(_text.substr(cluster_start, offset - cluster_start));
            cluster_start = offset;
        }
        prev = codepoint;
    });

    if (cluster_start != std::string_view::npos) {
        out.emplace_back(_text.substr(cluster_start));
    }
    return out;
}

std::vector<std::string_view> UnicodeProcessor::split_by_punctuation() const {
    if (_text.empty()) return {};

    std::vector<std::string_view> out;
    size_t chunk_start = std::string_view::npos;

    for_each_utf8_unit([&](size_t offset, size_t byte_len, char32_t codepoint) {
        if (is_sentence_punctuation(codepoint)) {
            if (chunk_start != std::string_view::npos) {
                const size_t end = offset + byte_len;
                out.emplace_back(_text.substr(chunk_start, end - chunk_start));
                chunk_start = std::string_view::npos;
            }
        } else if (chunk_start == std::string_view::npos) {
            chunk_start = offset;
        }
    });

    if (chunk_start != std::string_view::npos) {
        out.emplace_back(_text.substr(chunk_start));
    }
    if (out.empty()) out.emplace_back(_text);
    return out;
}

} // namespace lazyllm
