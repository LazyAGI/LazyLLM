#include "unicode_processor.hpp"

#include <utf8proc.h>

namespace lazyllm {

const std::array<char32_t, 6> UnicodeProcessor::kSentenceEndingCodepoints = {
    U'!',
    U'.',
    U'?',
    U'\u3002', // CJK full stop
    U'\uFF01', // fullwidth exclamation mark
    U'\uFF1F', // fullwidth question mark
};

const std::array<char32_t, 10> UnicodeProcessor::kSubSentencePunctuationCodepoints = {
    U'!',
    U',',
    U'.',
    U';',
    U'?',
    U'\u3002', // CJK full stop
    U'\uFF01', // fullwidth exclamation mark
    U'\uFF0C', // fullwidth comma
    U'\uFF1B', // fullwidth semicolon
    U'\uFF1F', // fullwidth question mark
};

template <typename Visitor>
void UnicodeProcessor::for_each_utf8_unit(Visitor&& visitor) const {
    size_t i = 0;
    auto text_size = _text.size();
    while (i < text_size) {
        int32_t codepoint = -1;
        const utf8proc_ssize_t n = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t*>(_text.data() + i),
            static_cast<utf8proc_ssize_t>(text_size - i),
            &codepoint);

        if (n <= 0) {
            i += 1;
            continue;
            // TODO: when adding stronger logging, collect all invalid UTF-8
            // bytes encountered and report them together.
        }

        visitor(i, static_cast<size_t>(n), codepoint);
        i += static_cast<size_t>(n);
    }
}

/**
 * UTF-8 text processing has three distinct layers:
 * 1) Byte: the storage unit in std::string_view; one code point uses 1-4 UTF-8 bytes.
 * 2) Code point: a Unicode scalar value (for example U+0061, U+4E2D), decoded by utf8proc_iterate.
 * 3) Grapheme cluster: one user-perceived character, which may contain multiple code points
 *    (for example base + combining mark, or emoji + VS/ZWJ sequences).
 *
 * This function splits by grapheme cluster, not by byte or code point:
 * - for_each_utf8_unit() uses utf8proc_iterate to decode UTF-8 and provide
 *   code point, byte offset, and byte length.
 * - utf8proc_grapheme_break_stateful(prev, codepoint, &state) determines whether
 *   there is a grapheme boundary between prev and the current code point.
 * - When a boundary appears, we emit a string_view slice over byte range
 *   [cluster_start, offset).
 *
 * This keeps splitting zero-copy (string_view) while following Unicode grapheme-boundary rules.
 */
std::vector<std::string_view> UnicodeProcessor::split_to_chars() const {
    std::vector<std::string_view> out;
    if (_text.empty()) return out;
    out.reserve(_text.size()); // Grapheme count <= byte length

    size_t cluster_start = std::string_view::npos;
    int32_t prev = -1;
    int32_t state = 0;

    for_each_utf8_unit([&](size_t offset, size_t byte_len, int32_t codepoint) {
        (void)byte_len;
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

// Sentence-ending punctuation is included at the end of each chunk.
// Any trailing text after the last sentence-ending punctuation is returned as the final chunk.
std::vector<std::string_view> UnicodeProcessor::split_by_sentence_endings() const {
    if (_text.empty()) return {};

    std::vector<std::string_view> out;
    size_t chunk_start = std::string_view::npos;
    bool trim_leading_space = true;

    for_each_utf8_unit([&](size_t offset, size_t byte_len, char32_t codepoint) {
        const bool is_space = utf8proc_category(codepoint) == UTF8PROC_CATEGORY_ZS
            || codepoint == U'\t' || codepoint == U'\n' || codepoint == U'\r' || codepoint == U'\f';
        if (chunk_start == std::string_view::npos && is_space && trim_leading_space) return;

        if (is_sentence_ending_punctuation(codepoint)) {
            if (chunk_start != std::string_view::npos) {
                const size_t end = offset + byte_len;
                out.push_back(_text.substr(chunk_start, end - chunk_start));
                chunk_start = std::string_view::npos;
                trim_leading_space = true;
            }
        } else if (chunk_start == std::string_view::npos) {
            chunk_start = offset;
            trim_leading_space = false;
        }
    });

    if (chunk_start != std::string_view::npos) {
        out.emplace_back(_text.substr(chunk_start));
    }
    if (out.empty()) out.emplace_back(_text);
    return out;
}

std::vector<std::string_view> UnicodeProcessor::split_by_punctuation() const {
    if (_text.empty()) return {};

    std::vector<std::string_view> out;
    size_t chunk_start = std::string_view::npos;
    bool trim_leading_space = true;

    for_each_utf8_unit([&](size_t offset, size_t byte_len, char32_t codepoint) {
        const bool is_space = utf8proc_category(codepoint) == UTF8PROC_CATEGORY_ZS
            || codepoint == U'\t' || codepoint == U'\n' || codepoint == U'\r' || codepoint == U'\f';
        if (chunk_start == std::string_view::npos && is_space && trim_leading_space) return;

        if (is_sub_sentence_punctuation(codepoint)) {
            if (chunk_start != std::string_view::npos) {
                const size_t end = offset + byte_len;
                out.push_back(_text.substr(chunk_start, end - chunk_start));
                chunk_start = std::string_view::npos;
                trim_leading_space = (
                    codepoint == U'.' || codepoint == U'!' || codepoint == U'\uFF1F'
                    || codepoint == U'\uFF01' || codepoint == U'?' || codepoint == U'\u3002');
            }
        } else if (chunk_start == std::string_view::npos) {
            chunk_start = offset;
            trim_leading_space = false;
        }
    });

    if (chunk_start != std::string_view::npos) {
        out.emplace_back(_text.substr(chunk_start));
    }
    if (out.empty()) out.emplace_back(_text);
    return out;
}

std::vector<std::string_view> UnicodeProcessor::split_to_n_parts(size_t n) const {
    if (_text.empty() || n == 0) return {};
    auto chars = split_to_chars();
    if (chars.empty()) return {_text};
    if (n >= chars.size()) return chars;

    std::vector<std::string_view> out;
    out.reserve(n);
    const size_t total = chars.size();
    const size_t base = total / n;
    const size_t rem = total % n;
    size_t idx = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t count = base + (i < rem ? 1 : 0);
        if (count == 0) continue;
        const size_t start_offset = chars[idx].data() - _text.data();
        const size_t end_offset = chars[idx + count - 1].data() - _text.data() + chars[idx + count - 1].size();
        out.emplace_back(_text.substr(start_offset, end_offset - start_offset));
        idx += count;
    }
    return out;
}

} // namespace lazyllm
