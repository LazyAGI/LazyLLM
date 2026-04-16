#include "text_splitter_base.hpp"
#include "unicode_processor.hpp"

namespace lazyllm {

/*
 * split_text
 * ----------
 * Purpose:
 * 1) Validate chunk budget after accounting for metadata tokens.
 * 2) Recursively split the original text view into token-bounded SplitUnit pieces.
 * 3) Merge the pieces into final chunk strings with overlap behavior aligned to Python implementation.
 *
 * Flow:
 * 1) Compute effective_chunk_size = chunk_size - metadata_size.
 * 2) Reject invalid/too-small budgets.
 * 3) Call split_recursive(...) to produce SplitUnit sequence.
 * 4) Call merge_chunks(...) to build final std::string chunks.
 *
 * Notes:
 * - This function returns std::string chunks intentionally because current tokenizer
 *   encode/decode materializes strings in the merge path.
 * - Ownership is explicit here to avoid dangling string_view in downstream DocNodeCore construction.
 *
 * TODO:
 * - After tokenizer supports true string_view encode/decode, migrate this path back to
 *   std::vector<std::string_view> and remove eager string materialization.
 */
std::vector<std::string> TextSplitterBase::split_text(std::string_view view, int metadata_size) const {
    if (view.empty()) return {""};
    int effective_chunk_size = _chunk_size - metadata_size;
    if (effective_chunk_size <= 0) {
        throw std::invalid_argument(
            "Metadata length (" + std::to_string(metadata_size) +
            ") is longer than chunk size (" + std::to_string(_chunk_size) +
            "). Consider increasing the chunk size or decreasing the size of your metadata to avoid this.");
    }
    else if (effective_chunk_size < 50) {
        // Keep Python behavior: this is only a warning there, not an exception.
        // We continue splitting with the small effective chunk size.
    }
    auto split_views = split_recursive(view, effective_chunk_size);
    std::vector<Chunk> splits;
    splits.reserve(split_views.size());
    for (const auto& split : split_views) {
        splits.push_back(Chunk{std::string(split.view), split.is_sentence, split.token_size});
    }
    return merge_chunks(std::move(splits), effective_chunk_size);
}

std::vector<ChunkView> TextSplitterBase::split_recursive(std::string_view view, const int chunk_size) const
{
    int token_size = get_token_size(view);
    if (token_size <= chunk_size) return {ChunkView{view, true, token_size}};

    auto [views, is_sentence] = split_by_functions(view);
    if (views.size() == 1) {
        int num_splits = (token_size + chunk_size - 1) / chunk_size;
        auto forced_views = UnicodeProcessor(view).split_to_n_parts(static_cast<size_t>(num_splits));
        std::vector<ChunkView> splits;
        splits.reserve(forced_views.size());
        for (const auto& v : forced_views) {
            splits.push_back({v, is_sentence, get_token_size(v)});
        }
        return splits;
    }

    std::vector<ChunkView> splits;
    for (const auto& segment_view : views) {
        const int seg_token_size = get_token_size(segment_view);
        if (seg_token_size == 0) continue;
        if (seg_token_size <= chunk_size) {
            splits.push_back({segment_view, is_sentence, seg_token_size});
        } else {
            auto new_splits = split_recursive(segment_view, chunk_size);
            splits.insert(splits.end(), new_splits.begin(), new_splits.end());
        }
    }
    return splits;
}

std::tuple<std::vector<std::string_view>, bool> TextSplitterBase::split_by_functions(std::string_view text) const
{
    auto views = split_text_while_keeping_separator(text, "\n\n\n");
    if (views.size() > 1) return {views, true};

    views = UnicodeProcessor(text).split_by_sentence_endings();
    if (views.size() > 1) return {views, true};

    views = UnicodeProcessor(text).split_by_punctuation();
    if (views.size() > 1) return {views, false};

    views = split_text_while_keeping_separator(text, " ");
    if (views.size() > 1) return {views, false};

    return {UnicodeProcessor(text).split_to_chars(), false};
}

std::vector<std::string_view> TextSplitterBase::split_text_while_keeping_separator(
    std::string_view text, std::string_view separator)
{
    if (text.empty()) return {};
    else if (separator.empty()) return {text};

    std::vector<std::string_view> result;
    size_t start = 0;
    const size_t sep_len = separator.size();
    while (start < text.size()) {
        const size_t idx = text.find(separator, start);
        if (idx == std::string_view::npos) {
            result.emplace_back(text.substr(start));
            break;
        }

        if (idx == start) {
            start += sep_len;
            continue;
        }

        result.emplace_back(text.substr(start, idx + sep_len - start));
        start = idx + sep_len;
    }
    return result;
}

/**
 *  @brief Build final chunks from token-sized split units while preserving overlap semantics.
 *
 *  @details
 *  1) Convert input SplitUnit views to owned strings (MergedSplit) for safe concatenation.
 *  2) If the tail split exactly matches chunk_size and overlap > 0:
 *     split it by token-halves via encode/decode, then push both halves back.
 *  3) Iterate backward:
 *     Add previous split, or part of it, to current split as overlap.
 *     - If the previous split is small enough, prepend it fully.
 *     - Otherwise, prepend token-based overlap suffix from previous split.
 *  4) Emit chunks in original order.
 *
 *  @todo Replace eager string materialization once tokenizer encode/decode supports
 *  end-to-end zero-copy string_view operations.
 */
std::vector<std::string> TextSplitterBase::merge_chunks(std::vector<Chunk> splits, int chunk_size) const
{
    if (splits.empty()) return {};

    if (splits.size() == 1) return {splits.front().text};

    if (splits.back().token_size == chunk_size && _overlap > 0) {
        Chunk end_split = splits.back();
        splits.pop_back();

        auto text_tokens = _tokenizer->encode(end_split.text);
        const size_t half = text_tokens.size() / 2;
        const auto split_it = text_tokens.begin() + static_cast<std::vector<int>::difference_type>(half);
        std::vector<int> prefix_tokens(text_tokens.begin(), split_it);
        std::vector<int> suffix_tokens(split_it, text_tokens.end());

        std::string prefix_text = _tokenizer->decode(prefix_tokens);
        std::string suffix_text = _tokenizer->decode(suffix_tokens);
        splits.push_back(
            Chunk{prefix_text, end_split.is_sentence, get_token_size(prefix_text)});
        splits.push_back(
            Chunk{suffix_text, end_split.is_sentence, get_token_size(suffix_text)});
    }

    Chunk end_split = splits.back();
    std::vector<std::string> reversed_result;
    reversed_result.reserve(splits.size());
    for (int idx = static_cast<int>(splits.size()) - 2; idx >= 0; --idx) {
        const Chunk& start_split = splits[static_cast<size_t>(idx)];
        if (start_split.token_size <= _overlap && end_split.token_size <= chunk_size - _overlap) {
            end_split = Chunk{
                start_split.text + end_split.text,
                start_split.is_sentence && end_split.is_sentence,
                start_split.token_size + end_split.token_size
            };
            continue;
        }

        if (end_split.token_size > chunk_size) {
            throw std::runtime_error("split token size is greater than chunk size.");
        }

        const int remaining_space = chunk_size - end_split.token_size;
        const int overlap_len = std::min({static_cast<int>(_overlap), remaining_space, start_split.token_size});
        if (overlap_len > 0) {
            auto start_tokens = _tokenizer->encode(start_split.text);
            std::vector<int> overlap_tokens(start_tokens.end() - overlap_len, start_tokens.end());
            std::string overlap_text = _tokenizer->decode(overlap_tokens);

            end_split = Chunk{
                overlap_text + end_split.text,
                end_split.is_sentence,
                end_split.token_size + overlap_len};
        }

        reversed_result.emplace_back(end_split.text);
        end_split = start_split;
    }

    reversed_result.emplace_back(end_split.text);
    std::reverse(reversed_result.begin(), reversed_result.end());
    return reversed_result;
}

}
