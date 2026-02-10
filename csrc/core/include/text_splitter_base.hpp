#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "doc_node.hpp"
#include "map_params.hpp"
#include "node_transform.hpp"
#include "tokenizer.hpp"

namespace lazyllm {

struct SplitUnit {
    std::string_view view;
    bool is_sentence = false;
    int token_size = 0;
};

class TextSplitterBase : public NodeTransform {
public:
    using SplitFn = std::function<std::vector<std::string>(const std::string&)>;

    explicit TextSplitterBase(
        std::optional<unsigned> chunk_size,
        std::optional<unsigned> overlap,
        std::optional<unsigned> worker_num,
        const std::string& encoding_name = "gpt2")
        : NodeTransform(_default_params.get_param_value<unsigned>("worker_num", worker_num, 0)),
          _chunk_size(_default_params.get_param_value<unsigned>("chunk_size", chunk_size, 1024)),
          _overlap(_default_params.get_param_value<unsigned>("overlap", overlap, 200))
    {
        if (_overlap > _chunk_size)
            throw std::runtime_error("'overlap' should be less than 'chunk_size'.");
        if (_chunk_size == 0)
            throw std::runtime_error("'chunk_size' should > 0");

        _tokenizer = std::make_shared<TiktokenTokenizer>(encoding_name);
    }

    std::vector<DocNode> transform(const DocNode* node) const override {
        if (node == nullptr) return {};
        return split_text(node->get_text_view(), get_node_metadata_size(node));
    }

    std::vector<DocNode> split_text(const std::string_view& view, int metadata_size) const {
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

    virtual void set_split_fns(
        const std::vector<SplitFn>& /*split_fns*/,
        const std::optional<std::vector<SplitFn>>& /*sub_split_fns*/ = std::nullopt) {}

    virtual void add_split_fn(const SplitFn& /*split_fn*/, const std::optional<size_t>& /*index*/ = std::nullopt) {}

    virtual void clear_split_fns() {}

protected:
    virtual std::vector<SplitUnit> split_recursive(const std::string_view& view, int chunk_size) const {
        int token_size = get_token_size(view);
        if (token_size <= chunk_size) return {SplitUnit{view, true, token_size}};

        auto [text_splits, is_sentence] = _get_splits_by_fns(view);
        std::vector<SplitUnit> results;
        for (const auto& segment : text_splits) {
            const int seg_token_size = get_token_size(segment);
            if (seg_token_size <= chunk_size) {
                results.push_back(SplitUnit{segment, is_sentence, seg_token_size});
            } else {
                auto sub_results = split_recursive(segment, chunk_size);
                results.insert(results.end(), sub_results.begin(), sub_results.end());
            }
        }
        return results;
    }

    std::vector<std::string> split_text_keep_separator(
        const std::string& text,
        const std::string& separator)
    {
        if (separator.empty()) return text.empty() ? std::vector<std::string>() : std::vector<std::string>{text};
        if (text.find(separator) == std::string::npos) return {text};

        std::vector<std::string> result;
        size_t start = 0;
        const size_t sep_len = separator.size();
        while (start < text.size()) {
            const size_t idx = text.find(separator, start);
            if (idx == std::string::npos) {
                result.emplace_back(text.substr(start));
                break;
            }
            if (idx == 0) {
                start = sep_len;
                continue;
            }
            result.emplace_back(text.substr(start, idx - start + sep_len));
            start = idx + sep_len;
        }
        return result;
    }


    virtual std::pair<std::vector<std::string>, bool> _get_splits_by_fns(const std::string& text) const
    {
        auto splits = split_text_keep_separator(text, "\n\n\n");
        if (splits.size() > 1) return {splits, true};

        splits = regex_find_all(text, R"([^.!?。？！]+[.!?。？！]?)");
        if (splits.size() > 1) return {splits, true};

        splits = regex_find_all(text, R"([^,.;。？！]+[,.;。？！]?)");
        if (splits.size() > 1) return {splits, false};

        splits = split_text_keep_separator(text, " ");
        if (splits.size() > 1) return {splits, false};

        return {split_to_chars(text), false};
    }

    int get_node_metadata_size(const DocNode* node) const {
        return std::max(
            get_token_size(node->get_metadata_string(MetadataMode::EMBED)),
            get_token_size(node->get_metadata_string(MetadataMode::LLM)));
    }

    int get_token_size(const std::string_view& view) const {
        return static_cast<int>(_tokenizer->encode(view).size());
    }

    static std::vector<std::string> regex_find_all(
        const std::string& text, const std::string& pattern)
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

    static std::vector<std::string> split_to_chars(const std::string& text) {
        std::vector<std::string> out;
        out.reserve(text.size());
        for (char c : text) out.emplace_back(1, c);
        return out;
    }

    std::vector<std::string> _merge(std::vector<SplitUnit> splits, int chunk_size) {
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

private:
    static MapParams _default_params;

protected:
    int _chunk_size;
    int _overlap;
    std::shared_ptr<Tokenizer> _tokenizer;
};

} // namespace lazyllm
