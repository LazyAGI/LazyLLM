#include "text_splitter_base.hpp"

#include <utf8proc.h>

namespace lazyllm {

std::vector<DocNode> TextSplitterBase::split_text(const std::string_view& view, int metadata_size) const {
    if (view.empty()) return {};
    int effective_chunk_size = _chunk_size - metadata_size;
    if (effective_chunk_size <= 0) {
        throw std::runtime_error(
            "Metadata length (" + std::to_string(metadata_size) +
            ") is longer than chunk size (" + std::to_string(_chunk_size) +
            "). Consider increasing the chunk size or decreasing the size of your metadata to avoid this.");
    }
    else if (effective_chunk_size < 50) {
        throw std::runtime_error(
            "Metadata length (" + std::to_string(metadata_size) + ") is close to chunk size (" +
            std::to_string(_chunk_size) + "). Resulting chunks are less than 50 tokens. " +
            "Consider increasing the chunk size or decreasing the size of " +
            "your metadata to avoid this.");
    }
    auto splits = split_recursive(view, effective_chunk_size);
    return _merge(splits, effective_chunk_size);
}

std::vector<SplitUnit> TextSplitterBase::split_recursive(
    const std::string_view& view, const int chunk_size) const
{
    int token_size = get_token_size(view);
    if (token_size <= chunk_size) return {SplitUnit{view, true, token_size}};

    auto [views, is_sentence] = split_by_functions(view);
    std::vector<SplitUnit> splits;
    for (const auto& view : views) {
        const int seg_token_size = get_token_size(view);
        if (seg_token_size <= chunk_size) {
            splits.emplace_back(view, is_sentence, seg_token_size);
        } else {
            auto new_splits = split_recursive(view, chunk_size);
            splits.insert(splits.end(), new_splits.begin(), new_splits.end());
        }
    }
    return splits;
}

std::tuple<std::vector<std::string_view>, bool> TextSplitterBase::split_by_functions(const std::string& text) const
{
    auto views = split_text_while_keeping_separator(text, "\n\n\n");
    if (views.size() > 1) return {views, true};

    views = regex_find_all(text, R"([^,.;。？！]+[,.;。？！]?)");
    if (views.size() > 1) return {views, false};

    views = split_text_while_keeping_separator(text, " ");
    if (views.size() > 1) return {views, false};

    return {split_to_chars(text), false};
}

std::vector<std::string_view> TextSplitterBase::split_text_while_keeping_separator(
    const std::string_view& text,
    const std::string_view& separator) const
{
    if (text.empty()) return {};
    else if (separator.empty()) return {text};

    std::vector<std::string_view> result;
    size_t start = 0;
    const size_t sep_len = separator.size();
    while (start <= text.size()) {
        const size_t idx = text.find(separator, start);
        const size_t end = (idx == std::string_view::npos) ? text.size() : idx;
        if (end > start) result.emplace_back(text.substr(start, end - start));
        if (idx == std::string_view::npos) break;
        start = idx + sep_len;
    }
    return result;
}

std::vector<std::string> TextSplitterBase::regex_find_all(
    const std::string_view& text,
    const std::string_view& pattern)
{
    std::regex re(pattern);
    std::vector<std::string> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
            it != std::sregex_iterator(); ++it) {
        out.emplace_back(it->str());
    }
    if (out.empty()) out.emplace_back(text);
    return out;
}

std::vector<std::string_view> TextSplitterBase::split_to_chars(const std::string_view& text) {
    std::vector<std::string_view> out;
    if (text.empty()) return out;
    out.reserve(text.size());

    const size_t len = text.size();
    size_t cluster_start = 0;
    size_t i = 0;
    utf8proc_int32_t prev = -1;
    utf8proc_propval_t state = 0;

    while (i < len) {
        utf8proc_int32_t codepoint = -1;
        const utf8proc_ssize_t n = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t*>(text.data() + i),
            static_cast<utf8proc_ssize_t>(len - i),
            &codepoint);
        if (n <= 0) {
            if (i > cluster_start) {
                out.emplace_back(text.substr(cluster_start, i - cluster_start));
            }
            out.emplace_back(text.substr(i, 1));
            i += 1;
            cluster_start = i;
            prev = -1;
            state = 0;
            continue;
        }

        if (prev >= 0 && utf8proc_grapheme_break_stateful(prev, codepoint, &state)) {
            out.emplace_back(text.substr(cluster_start, i - cluster_start));
            cluster_start = i;
        }
        prev = codepoint;
        i += static_cast<size_t>(n);
    }
    if (cluster_start < len) {
        out.emplace_back(text.substr(cluster_start, len - cluster_start));
    }
    return out;
}

std::vector<std::string> TextSplitterBase::_merge(std::vector<SplitUnit> splits, int chunk_size) {
    if (splits.empty()) return {};
    if (splits.size() == 1) return {splits.front().text};

    SplitUnit end_split = splits.back();
    if (end_split.token_size == chunk_size && _overlap > 0) {
        splits.pop_back();
        auto text_tokens = encode(end_split.text);
        const size_t half = text_tokens.size() / 2;
        std::vector<int> p_tokens(text_tokens.begin(), text_tokens.begin() + half);
        std::vector<int> n_tokens(text_tokens.begin() + half, text_tokens.end());
        std::string p_text = decode(p_tokens);
        std::string n_text = decode(n_tokens);
        splits.push_back(SplitUnit{p_text, end_split.is_sentence, get_token_size(p_text)});
        splits.push_back(SplitUnit{n_text, end_split.is_sentence, get_token_size(n_text)});
        end_split = splits.back();
    }

    std::vector<std::string> result;
    for (int idx = static_cast<int>(splits.size()) - 2; idx >= 0; --idx) {
        SplitUnit start_split = splits[static_cast<size_t>(idx)];
        if (start_split.token_size <= _overlap &&
            end_split.token_size <= chunk_size - _overlap) {
            const bool is_sentence = start_split.is_sentence && end_split.is_sentence;
            const int token_size = start_split.token_size + end_split.token_size;
            end_split = SplitUnit{start_split.text + end_split.text, is_sentence, token_size};
            continue;
        }

        if (end_split.token_size > chunk_size) {
            throw std::runtime_error("split token size is greater than chunk size.");
        }

        const int remaining_space = chunk_size - end_split.token_size;
        const int overlap_len = std::min({_overlap, remaining_space, start_split.token_size});
        if (overlap_len > 0) {
            auto start_tokens = encode(start_split.text);
            std::vector<int> overlap_tokens(
                start_tokens.end() - overlap_len, start_tokens.end());
            std::string overlap_text = decode(overlap_tokens);
            end_split = SplitUnit{overlap_text + end_split.text, end_split.is_sentence,
                                end_split.token_size + overlap_len};
        }

        result.insert(result.begin(), end_split.text);
        end_split = start_split;
    }

    result.insert(result.begin(), end_split.text);
    return result;
}

}
