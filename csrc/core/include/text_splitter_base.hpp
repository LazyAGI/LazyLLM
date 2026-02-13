#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
    static MapParams _default_params;

    explicit TextSplitterBase(
        std::optional<unsigned> chunk_size,
        std::optional<unsigned> overlap,
        std::optional<unsigned> worker_num,
        const std::string& encoding_name = "gpt2")
        : NodeTransform(_default_params.get_param_value<unsigned>("worker_num", worker_num, 0)),
          _chunk_size(_default_params.get_param_value<unsigned>("chunk_size", chunk_size, 1024)),
          _overlap(_default_params.get_param_value<unsigned>("overlap", overlap, 200))
    {
        if (_overlap > _chunk_size) throw std::runtime_error("'overlap' should be less than 'chunk_size'.");
        if (_chunk_size == 0) throw std::runtime_error("'chunk_size' should > 0");

        _tokenizer = std::make_shared<TiktokenTokenizer>(encoding_name);
    }

    std::vector<DocNode> transform(const DocNode* node) const override {
        if (node == nullptr) return {};
        auto chunks = split_text(node->get_text_view(), get_node_metadata_size(node));
        std::vector<DocNode> nodes;
        nodes.reserve(chunks.size());
        for (const auto& chunk : chunks) {
            DocNode chunk_node(chunk);
            chunk_node.set_root_text(std::string(chunk));
            nodes.emplace_back(std::move(chunk_node));
        }
        return nodes;
    }

    std::vector<std::string> split_text(const std::string_view& view, int metadata_size) const;
    static std::vector<std::string_view> split_text_while_keeping_separator(
        const std::string_view& text,
        const std::string_view& separator);

    TextSplitterBase& from_tiktoken_encoder(
        const std::string& encoding_name = "gpt2",
        const std::optional<std::string>& model_name = std::nullopt)
    {
        _tokenizer = std::make_shared<TiktokenTokenizer>(model_name.value_or(encoding_name));
        return *this;
    }

    void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer) { _tokenizer = std::move(tokenizer); }

    virtual void set_split_functions(
        const std::vector<SplitFn>&,
        const std::optional<std::vector<SplitFn>>& = std::nullopt) {}
    virtual void add_split_function(const SplitFn&, const std::optional<size_t>& = std::nullopt) {}
    virtual void clear_split_functions() {}

protected:
    virtual std::vector<SplitUnit> split_recursive(const std::string_view& view, const int chunk_size) const;
    virtual std::vector<std::string> merge_chunks(const std::vector<SplitUnit>& splits, int chunk_size) const;

private:
    std::tuple<std::vector<std::string_view>, bool> split_by_functions(const std::string_view& text) const;

    int get_node_metadata_size(const DocNode* node) const {
        return std::max(
            get_token_size(node->get_metadata_string(MetadataMode::EMBED)),
            get_token_size(node->get_metadata_string(MetadataMode::LLM)));
    }

    int get_token_size(const std::string_view& view) const {
        return static_cast<int>(_tokenizer->encode(view).size());
    }

protected:
    int _chunk_size;
    int _overlap;
    std::shared_ptr<Tokenizer> _tokenizer;
};

} // namespace lazyllm
