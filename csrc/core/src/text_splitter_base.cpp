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
 * - Ownership is explicit here to avoid dangling string_view in downstream DocNode construction.
 *
 * TODO:
 * - After tokenizer supports true string_view encode/decode, migrate this path back to
 *   std::vector<std::string_view> and remove eager string materialization.
 */
std::vector<std::string> TextSplitterBase::split_text(const std::string_view& view, int metadata_size) const {
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
    return merge_chunks(splits, effective_chunk_size);
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

std::tuple<std::vector<std::string_view>, bool> TextSplitterBase::split_by_functions(const std::string_view& text) const
{
    auto views = split_text_while_keeping_separator(text, "\n\n\n");
    if (views.size() > 1) return {views, true};

    views = UnicodeProcessor(text).split_by_punctuation();
    if (views.size() > 1) return {views, false};

    views = split_text_while_keeping_separator(text, " ");
    if (views.size() > 1) return {views, false};

    return {UnicodeProcessor(text).split_to_chars(), false};
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
std::vector<std::string> TextSplitterBase::merge_chunks(const std::vector<SplitUnit>& splits, int chunk_size) const {
    if (splits.empty()) return {};

    struct MergedSplit {
        std::string text;
        bool is_sentence = false;
        int token_size = 0;

        MergedSplit& operator+=(const MergedSplit& r) {
            text += r.text;
            is_sentence = is_sentence && r.is_sentence;
            token_size += r.token_size;
            return *this;
        }
    };
    std::vector<MergedSplit> merged_splits;
    merged_splits.reserve(splits.size() + 2);
    for (const auto& split : splits)
        merged_splits.push_back(MergedSplit{std::string(split.view), split.is_sentence, split.token_size});

    if (merged_splits.size() == 1) return {merged_splits.front().text};

    if (merged_splits.back().token_size == chunk_size && _overlap > 0) {
        MergedSplit end_split = merged_splits.back();
        merged_splits.pop_back();

        auto text_tokens = _tokenizer->encode(end_split.text);
        const size_t half = text_tokens.size() / 2;
        const auto split_it = text_tokens.begin() + static_cast<std::vector<int>::difference_type>(half);
        std::vector<int> prefix_tokens(text_tokens.begin(), split_it);
        std::vector<int> suffix_tokens(split_it, text_tokens.end());

        std::string prefix_text = _tokenizer->decode(prefix_tokens);
        std::string suffix_text = _tokenizer->decode(suffix_tokens);
        merged_splits.push_back(
            MergedSplit{prefix_text, end_split.is_sentence, get_token_size(prefix_text)});
        merged_splits.push_back(
            MergedSplit{suffix_text, end_split.is_sentence, get_token_size(suffix_text)});
    }

    MergedSplit end_split = merged_splits.back();
    std::vector<std::string> reversed_result;
    reversed_result.reserve(merged_splits.size());
    for (auto idx = merged_splits.size() - 2; idx >= 0; --idx) {
        const MergedSplit& start_split = merged_splits[idx];
        if (start_split.token_size <= _overlap && end_split.token_size <= chunk_size - _overlap) {
            end_split += start_split;
            continue;
        }

        if (end_split.token_size > chunk_size) {
            throw std::runtime_error("split token size is greater than chunk size.");
        }

        const int remaining_space = chunk_size - end_split.token_size;
        const int overlap_len = std::min({_overlap, remaining_space, start_split.token_size});
        if (overlap_len > 0) {
            auto start_tokens = _tokenizer->encode(start_split.text);
            std::vector<int> overlap_tokens(start_tokens.end() - overlap_len, start_tokens.end());
            std::string overlap_text = _tokenizer->decode(overlap_tokens);

            end_split = MergedSplit{
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
